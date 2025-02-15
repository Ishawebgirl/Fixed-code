import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import Cookies from 'js-cookie'
import { useNavigate } from 'react-router-dom'
import * as jwtDecode from 'jwt-decode'
import * as Model from 'api/model'
import configSettings from 'settings/config.json'
import {
  AccountAuthentication,
  AccountAuthenticationRefresh,
  AccountClient,
  GoogleAccountAuthentication,
} from 'api/model'
import { Identity } from 'api/identity'
import { Log } from 'helpers/log'
import { max } from 'lodash'
import { CookieName, LocalStorageKey } from 'helpers/constants'

var authTimer: ReturnType<typeof setTimeout>
var authTimerCountdown: ReturnType<typeof setInterval>

interface IAuthenticationContext {
  redirectUnauthenticated: (includeRedirectParam: boolean) => void
  authorize: (
    emailAddress: string,
    password: string
  ) => Promise<Model.AuthenticationToken | null>
  authorizeGoogle: (
    credential: string,
    nonce: string
  ) => Promise<Model.AuthenticationToken | null>
  clearIdentity: () => void
  identity: Identity | null
  getProvider: () => string
  updateAvatarUrl: (file: Model.FileParameter) => void
  jwtAccessTokenLifeRemaining: number
}

export const AuthenticationContext = (): IAuthenticationContext => {
  const {
    redirectUnauthenticated,
    authorize,
    authorizeGoogle,
    clearIdentity,
    identity,
    getProvider,
    updateAvatarUrl,
    jwtAccessTokenLifeRemaining,
  } = useContext(Context)

  return {
    redirectUnauthenticated,
    authorize,
    authorizeGoogle,
    clearIdentity,
    identity,
    getProvider,
    updateAvatarUrl,
    jwtAccessTokenLifeRemaining,
  }
}

const Context = createContext({} as IAuthenticationContext)

export function AuthenticationProvider({ children }: { children: any }) {
  const [jwtAccessTokenLifeRemaining, setJwtAccessTokenLifeRemaining] = useState(100)
  const [identity, setIdentity] = useState(null as Identity | null)

  const accountApi = useMemo(() => new AccountClient(process.env.REACT_APP_API_URL), [])
  const navigate = useRef(useNavigate())

  const redirectUnauthenticated = (includeRedirectParam: boolean) => {
    if (includeRedirectParam && window.location.pathname.indexOf('/signin') < 0) {
      navigate.current(
        '/signin?redirectTo=' + encodeURIComponent(window.location.pathname)
      )
    } else {
      navigate.current('/signin')
    }
  }

  const getProvider = (): string => {
    const provider = Cookies.get(CookieName.provider)

    if (provider && provider.length > 0) {
      return provider
    } else {
      return ''
    }
  }

  const saveProvider = (provider: string) => {
    const params = { domain: window.location.hostname, secure: true, expires: 365 }
    Cookies.set(CookieName.provider, provider, params)
  }

  const getJwtTokenTtl = (expiration: number): number => {
    let margin =
      (Number(configSettings.AccessTokenTimeout) *
        Number(configSettings.AccessTokenRefreshMarginPercent)) /
      100

    margin = max([margin, Number(configSettings.AccessTokenMinimumRefreshMargin)])!

    const ttl = expiration - margin * 1000 - new Date().getTime()
    return ttl
  }

  const clearIdentity = () => {
    clearTimeout(authTimer)
    clearInterval(authTimerCountdown)
    localStorage.removeItem(LocalStorageKey.auth)
  }

  const authorize = async (
    emailAddress: string,
    password: string
  ): Promise<Model.AuthenticationToken | null> => {
    Log.clear()
    Log.add('authorizing...')

    let token: Model.AuthenticationToken | null = null
    const request = AccountAuthentication.fromJS({
      emailAddress: emailAddress,
      password: password,
    })

    await accountApi
      .authenticate(request)
      .then(async (result) => {
        saveIdentity(
          result.role,
          emailAddress,
          result.avatarUrl,
          result.niceName,
          result.validUntil.toISOString()
        )

        saveProvider('Local')

        token = result
      })
      .catch((error) => {
        throw error
      })

    return token
  }

  const authorizeGoogle = async (
    credential: string,
    nonce: string
  ): Promise<Model.AuthenticationToken | null> => {
    Log.clear()
    Log.add('authorizing google...')

    let token: Model.AuthenticationToken | null = null
    const item = jwtDecode.jwtDecode<{ email: string; nonce: string }>(credential)

    const request = GoogleAccountAuthentication.fromJS({
      emailAddress: item.email,
      googleJwt: credential,
    })

    await accountApi
      .authenticateGoogle(request)
      .then(async (result) => {
        if (nonce !== item.nonce) {
          clearIdentity()
          // eslint-disable-next-line no-throw-literal
          throw {
            response: {
              data: {
                errorCode: 2201,
                errorCodeName: Model.ErrorCode.GoogleJwtNonceInvalid,
              },
            },
          }
        }

        saveIdentity(
          result.role,
          item.email,
          result.avatarUrl,
          result.niceName,
          result.validUntil.toISOString()
        )
        saveProvider('Google')

        token = result
      })
      .catch((error) => {
        throw error
      })

    return token
  }

  const restartJwtTimers = useCallback(
    (identity: Identity) => {
      clearTimeout(authTimer)
      clearInterval(authTimerCountdown)

      const ttl = getJwtTokenTtl(Date.parse(identity.expiration))

      authTimer = setTimeout(() => {
        reauthorize(identity.emailAddress)
      }, ttl)

      authTimerCountdown = setInterval(() => {
        const countdown = (Date.parse(identity.expiration) - new Date().getTime()) / 1000

        setJwtAccessTokenLifeRemaining(
          (100.0 * countdown) / Number(configSettings.AccessTokenTimeout)
        )
        localStorage.setItem(LocalStorageKey.auth, countdown.toString())
      }, 1000)
    },
    // issue with param precedence
    // eslint-disable-next-line
    [setJwtAccessTokenLifeRemaining]
  )

  const updateAvatarUrl = async (file: Model.FileParameter) => {
    try {
      var url = await accountApi.uploadAvatar(file)
      setIdentity(
        new Identity(
          identity!.role,
          identity!.emailAddress,
          url + '?t=' + Date.now(),
          identity!.niceName,
          identity!.nonce,
          identity!.expiration
        )
      )
    } catch (error: any) {
      Log.add(JSON.stringify(error))
    }
  }

  const saveIdentity: (
    role: string,
    emailAddress: string,
    avatarUrl: string,
    niceName: string,
    validUntil: string
  ) => void = useCallback(
    (
      role: string,
      emailAddress: string,
      avatarUrl: string,
      niceName: string,
      validUntil: string
    ) => {
      const emailAddresses = emailAddress.split(':')

      const updatedIdentity = new Identity(
        role,
        emailAddresses[0],
        avatarUrl,
        niceName,
        null,
        validUntil
      )
      setIdentity(updatedIdentity)
      persistIdentity(updatedIdentity)
      restartJwtTimers(updatedIdentity)
    },
    [restartJwtTimers]
  )

  const persistIdentity = (updatedIdentity: Identity) => {
    const expiresInSeconds =
      (new Date(updatedIdentity.expiration).getTime() - Date.now()) / 1000
    const expiresInDays = expiresInSeconds / 60 / 60 / 24
    const params = {
      domain: window.location.hostname,
      secure: true,
      expires: expiresInDays,
    }

    Cookies.set(CookieName.identity, JSON.stringify(updatedIdentity), params)
  }

  const reauthorize = useCallback(
    async (emailAddress: string): Promise<Model.AuthenticationToken | null> => {
      Log.add('reauthorizing...')

      const request = AccountAuthenticationRefresh.fromJS({
        emailAddress: emailAddress,
        refreshToken: '_',
      })

      await accountApi
        .authenticateRefresh(request)
        .then(async (result) => {
          saveIdentity(
            result.role,
            emailAddress,
            result.avatarUrl,
            result.niceName,
            result.validUntil.toISOString()
          )
          return result
        })
        .catch((error) => {
          throw error
        })

      return null
    },
    // eslint-disable-next-line
    [accountApi]
  )

  useEffect(() => {
    const identityCookie = Cookies.get(CookieName.identity)

    if (identityCookie !== undefined) {
      const identity = Identity.parse(identityCookie)

      if (identity) {
        setIdentity(identity)
      }
    }
  }, [])

  useEffect(() => {
    // on page refresh we need to restart the JWT token timer
    const jwtCountdown = Number(localStorage.getItem(LocalStorageKey.auth))

    if (
      jwtAccessTokenLifeRemaining === 100 &&
      identity?.emailAddress != null &&
      !isNaN(jwtCountdown)
    ) {
      restartJwtTimers(identity)
    }
  }, [jwtAccessTokenLifeRemaining, restartJwtTimers, identity])

  return (
    <Context.Provider
      value={{
        redirectUnauthenticated,
        authorize,
        authorizeGoogle,
        clearIdentity,
        identity,
        getProvider,
        updateAvatarUrl,
        jwtAccessTokenLifeRemaining,
      }}>
      {children}
    </Context.Provider>
  )
}

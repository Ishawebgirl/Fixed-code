import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import IdlePopup from 'components/IdlePopup'
import NavBar from 'components/NavBar'
import { HelmetProvider } from 'react-helmet-async'
import configSettings from 'settings/config.json'
import { Outlet } from 'react-router-dom'
import Avatar from 'components/Avatar'
import { Popover } from 'react-tiny-popover'
import Icon from 'components/Icon'
import { AuthenticationContext } from 'contexts/AuthenticationContext'
import { Log } from 'helpers/log'
import useNotification from 'hooks/useNotification'
import * as Model from 'api/model'

interface IAuthenticatedLayout {
  header?: any
}

export const AuthenticatedLayout = ({ header }: IAuthenticatedLayout) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isIdlePopupOpen, setIsIdlePopupOpen] = useState(false)
  const [lastActiveTime, setLastActiveTime] = useState(Date.now())
  const [idleTimeoutSecondsRemaining, setIdleTimeoutSecondsRemaining] = useState(
    configSettings.IdleTimeout
  )
  const [theme, setTheme] = useState('light')
  const [isPopoverOpen, setIsPopoverOpen] = useState(false)

  const {
    redirectUnauthenticated,
    jwtAccessTokenLifeRemaining,
    clearIdentity,
    updateAvatarUrl,
    identity,
  } = AuthenticationContext()

  const domEvents = useMemo(() => ['click', 'scroll', 'keypress'], [])
  //const domEvents = ["click", "scroll", "keypress", "mousemove"];

  const mainRef = useRef<HTMLDivElement>(null)
  const avatarUploadRef = useRef<HTMLInputElement | null>(null)

  const { notify } = useNotification()

  let idleTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const signout = useCallback(
    (includeRedirectParam: boolean) => {
      clearIdentity()

      if (idleTimer.current) {
        clearInterval(idleTimer.current)
      }
      redirectUnauthenticated(includeRedirectParam)
    },
    [clearIdentity, redirectUnauthenticated]
  )

  const resetIdleTimer = useCallback(() => {
    const now = Date.now()
    setLastActiveTime(now)

    if (idleTimer.current) {
      clearInterval(idleTimer.current)
    }
    idleTimer.current = setInterval(() => {
      const secondsRemaining =
        Number(configSettings.IdleTimeout) - Math.round((Date.now() - now) / 1000)

      setIdleTimeoutSecondsRemaining(secondsRemaining + configSettings.IdlePopupDuration)

      if (secondsRemaining + configSettings.IdlePopupDuration <= 0) {
        signout(true)
      }
    }, 1000)
  }, [signout])

  const onIdlePopupClose = useCallback(
    (isLogout: boolean) => {
      setIsIdlePopupOpen(false)

      if (idleTimer.current) {
        clearInterval(idleTimer.current)
        idleTimer.current = null
      }

      if (isLogout) {
        clearIdentity()
        redirectUnauthenticated(true)
      } else {
        setLastActiveTime(Date.now())
      }
    },
    [clearIdentity, redirectUnauthenticated]
  )

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    localStorage.setItem('theme', newTheme)
    setTheme(newTheme)
  }

  const bodyClickEventListener = useCallback((event: MouseEvent) => {
    const popoverRef = document.querySelector(
      '.auth-account .popover-container'
    ) as HTMLElement

    if (popoverRef && !popoverRef.contains(event.target as Node)) {
      setIsPopoverOpen(false)
      document.body.removeEventListener('click', bodyClickEventListener)
      popoverRef?.classList?.add('closing')
    }
  }, [])

  useEffect(() => {
    return () => {
      document.body.removeEventListener('click', bodyClickEventListener)
    }
  }, [bodyClickEventListener])

  const toggleIsPopoverOpen = () => {
    if (!isPopoverOpen) {
      setIsPopoverOpen(true)
      setTimeout(() => {
        document.body.addEventListener('click', bodyClickEventListener)
      }, 200)
    }
  }

  const showLog = () => {
    let entries = Log.get() ?? ''
    entries = '<p>' + entries.replace(/\r\n/g, '</p><p>') + '</p>'
    notify('Log', entries, 'log')
  }

  const resizeObserver = new ResizeObserver((entries) => {
    // outer divs don't resize properly, leaving empty space at the bottom (only visible in dark mode)
    for (const entry of entries) {
      if (entry.contentBoxSize) {
        const contentBoxSize = entry.target.scrollHeight + 12
        mainRef.current?.style.setProperty('height', contentBoxSize + 'px')
      }
    }
  })

  const handleAvatarChange = async (event: any) => {
    const file = event.target.files[0]

    if (file) {
      const fileParameter: Model.FileParameter = {
        data: file,
        fileName: file.name ?? 'Unknown',
      }

      updateAvatarUrl(fileParameter)
    }

    avatarUploadRef.current!.value = ''
  }

  useEffect(() => {
    const id = setTimeout(
      () => {
        setIsIdlePopupOpen(true)
      },
      Number(configSettings.IdleTimeout) * 1000
    )

    return clearTimeout.bind(null, id)
  }, [lastActiveTime])

  useEffect(() => {
    domEvents.forEach((event: any) => document.addEventListener(event, resetIdleTimer))

    return () => {
      domEvents.forEach((event: any) =>
        document.removeEventListener(event, resetIdleTimer)
      )
    }
  }, [domEvents, resetIdleTimer])

  useEffect(() => {
    const elem = document.querySelector('#inner')

    if (elem) {
      resizeObserver.observe(elem)
    }

    if (idleTimer.current) {
      clearInterval(idleTimer.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (identity !== null) {
      setIsAuthenticated(true)
      resetIdleTimer()
    } else {
      // setIsAuthenticated(false)
      // redirectUnauthenticated(true)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [identity])

  useEffect(() => {
    const storedTheme = localStorage.getItem('theme')?.toString() ?? 'light'

    if (storedTheme) {
      setTheme(storedTheme)
    }
  }, [])

  return (
    <>
      <HelmetProvider>
        <title>Reclaim</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
        <link rel="stylesheet" type="text/css" href="/styles/global.scss" />
      </HelmetProvider>
      <div
        data-theme={theme}
        className={`auth-container ${isAuthenticated ? '' : 'not-authenticated'}`}>
        <IdlePopup
          isOpen={isIdlePopupOpen}
          onClose={onIdlePopupClose}
          remainingTime={idleTimeoutSecondsRemaining}></IdlePopup>
        <div className="auth-timeout-bar">
          <div style={{ width: `${jwtAccessTokenLifeRemaining}%` }}></div>
        </div>
        <div className="outer">
          <NavBar role={identity?.role ?? ''}></NavBar>
          <main ref={mainRef}>
            <div id="overlay" className="wrapper">
              <div className="auth-account">
                {header}
                <Icon className="button" name="Inbox">
                  <span className="icon-badge">3</span>
                </Icon>
                <Popover
                  parentElement={
                    document.getElementsByClassName('auth-account')[0] as HTMLElement
                  }
                  containerClassName="popover-container"
                  isOpen={isPopoverOpen}
                  content={
                    <div className={`popover-content`}>
                      <Avatar
                        url={identity?.avatarUrl ?? ''}
                        name={identity?.niceName ?? ''}
                        onClick={(e) => {
                          e.stopPropagation()
                          avatarUploadRef.current?.click()
                        }}></Avatar>
                      <input
                        type="file"
                        accept=".jpg,.png"
                        ref={avatarUploadRef}
                        onChange={handleAvatarChange}
                        value={''}
                        style={{ visibility: 'hidden', width: '0' }}
                      />
                      <span className="name">{identity?.niceName ?? ''}</span>
                      <span className="role">
                        <span
                          className={`role-name property-tag ${identity?.role ?? ''}`}>
                          {identity?.role ?? ''}
                        </span>
                      </span>
                      <hr></hr>
                      <div>
                        <Icon name="User" />
                        Profile
                      </div>
                      <div
                        onClick={() => {
                          toggleTheme()
                        }}>
                        {theme === 'light' ? (
                          <div>
                            <Icon name="Moon" />
                            Dark mode
                          </div>
                        ) : (
                          <div>
                            <Icon name="Sun" />
                            Light mode
                          </div>
                        )}
                      </div>
                      <div>
                        <Icon name="Gear" />
                        Settings
                      </div>
                      <div onClick={showLog}>
                        <Icon name="BookOpen" />
                        Session log
                      </div>
                      <div onClick={() => signout(false)}>
                        <Icon name="SignOut" />
                        Sign out
                      </div>
                    </div>
                  }
                  positions={['left', 'right']}>
                  <Avatar
                    url={identity?.avatarUrl ?? ''}
                    name={identity?.niceName ?? ''}
                    onClick={toggleIsPopoverOpen}></Avatar>
                </Popover>
              </div>
              <div id="inner">
                <Outlet />
              </div>
            </div>
          </main>
        </div>
      </div>
    </>
  )
}

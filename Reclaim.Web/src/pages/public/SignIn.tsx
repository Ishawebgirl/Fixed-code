import TextInput from 'components/TextInput'
import { useEffect, useRef, useState } from 'react'
import { UnauthenticatedLayout } from 'layouts/UnauthenticatedLayout'
import { RegexPattern } from 'helpers/constants'
import { AuthenticationContext } from 'contexts/AuthenticationContext'
import * as Model from 'api/model'
import { GoogleLogin } from '@react-oauth/google'
import { v4 } from 'uuid'
import { Link, useNavigate } from 'react-router-dom'
import { Log } from 'helpers/log'

const SignIn = () => {
  const [nonce] = useState(v4())
  const [emailAddress, setEmailAddress] = useState('')
  const [password, setPassword] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [isSubmitButtonEnabled, setIsSubmitButtonEnabled] = useState(false)
  const [isPasswordResetLinkVisible, setIsPasswordResetLinkVisible] = useState(false)

  const { authorize, authorizeGoogle, clearIdentity, getProvider } =
    AuthenticationContext()

  const navigate = useRef(useNavigate())

  useEffect(() => {
    Log.add('api url: ' + process.env.REACT_APP_API_URL)
  }, [])

  useEffect(() => {
    clearIdentity()
    const params = new URLSearchParams(window.location.search)
    const provider = getProvider()

    if (provider === 'Local') {
      setEmailAddress(params.get('emailAddress') ?? '')
    }
  }, [clearIdentity, getProvider])

  useEffect(() => {
    setIsSubmitButtonEnabled(emailAddress.length > 0 && password.length > 0)
  }, [emailAddress, password])

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()

    if (!validate()) {
      return
    }

    await signIn(async (): Promise<Model.AuthenticationToken | null> => {
      const result = await authorize(emailAddress, password)
      return result
    })
  }

  const handleGoogleSubmit = async (credentialResponse: any) => {
    if (credentialResponse == null || credentialResponse.credential == null) {
      setErrorMessage(
        'An error occurred while attempting to sign in via Google: the remote system failed to return a valid credential token.'
      )
      return
    }

    await signIn(async (): Promise<Model.AuthenticationToken | null> => {
      return await authorizeGoogle(credentialResponse.credential, nonce)
    })
  }

  const handleGoogleError = async () => {
    setErrorMessage('An error occurred while attempting to sign in via Google.')
  }

  const validate = () => {
    if (emailAddress.length === 0 || password.length === 0) {
      setErrorMessage(' ')
      return false
    }

    let isValid = false
    const emailAddresses = emailAddress.split(':')

    if (emailAddresses.length === 2) {
      if (
        new RegExp(RegexPattern.emailAddress).test(emailAddresses[0]) &&
        new RegExp(RegexPattern.emailAddress).test(emailAddresses[1])
      ) {
        isValid = true
      }
    } else if (new RegExp(RegexPattern.emailAddress).test(emailAddress)) {
      isValid = true
    }

    if (!isValid) {
      setErrorMessage('The email address you provided is not valid.')
    }

    return isValid
  }

  const signIn = async (
    authFunction: () => Promise<Model.AuthenticationToken | null>
  ) => {
    try {
      const token = await authFunction()

      if (token === null) {
        setErrorMessage(
          'The request could not be completed, an authentication token was not returned from the API.'
        )
        return
      }

      const params = new URLSearchParams(window.location.search)
      const redirectTo = params.get('redirectTo')
      const urlParts = redirectTo ? redirectTo.split('/') : ['']

      if (redirectTo && urlParts.length > 0) {
        navigate.current(decodeURIComponent(redirectTo))
      } else {
        navigate.current(`/dashboard`)
      }
    } catch (error: any) {
      Log.add(JSON.stringify(error))

      if (error instanceof TypeError && error.message === 'Failed to fetch') {
        setErrorMessage(
          'The request could not be completed, the backend API may not be configured correctly.'
        )
        return
      }

      const apiError = JSON.parse(error.response)

      switch (apiError?.errorCodeName) {
        case Model.ErrorCode.AccountCredentialsInvalid:
          setErrorMessage(
            'The credentials you provided are invalid.  Please check your email address and password and try again to sign in.'
          )
          setIsPasswordResetLinkVisible(true)
          break

        case Model.ErrorCode.AccountExternalCredentialsInvalid:
          setErrorMessage(
            'The Google account does not appear to be configured as an account on this system.  Please try signing in with an email address and password, or create a new account on the registration page.'
          )
          break

        case Model.ErrorCode.AccountEmailAddressNotConfirmed:
          setErrorMessage(
            'You cannot sign in until you have confirmed your account.  Please check your email for the welcome message we sent when you registered.<br/><br/>If you have not received the welcome message, please use the link below to request a password reset.'
          )
          break

        case Model.ErrorCode.AccountLockedOut:
        case Model.ErrorCode.AccountLockedOutOverride:
          setErrorMessage(
            'Your account has been locked due to too many failed sign-in attempts.  Please wait for 30 minutes, then request a password reset.'
          )
          break

        case Model.ErrorCode.AccountTombstoned:
          setErrorMessage('Your account has been disabled.')
          break

        case Model.ErrorCode.AccountCredentialsExpired:
        case Model.ErrorCode.AccountCredentialsNotConfirmed:
          setErrorMessage(
            'Your account credentials are expired.  Please check your email for a password reset email, or click the Forgot Password link below to retry.'
          )
          break

        case Model.ErrorCode.AccountRequiresIdentityProviderLocal:
          setErrorMessage(
            'You created your account using an email address and password.  Please sign in using these credentials rather than using the Google sign-in framework.'
          )
          break

        case Model.ErrorCode.AccountRequiresIdentityProviderGoogle:
          setErrorMessage(
            'You created your account using your Google account, rather than using a standard email address and password.  Please sign in using the Google sign-in button.'
          )
          break

        case Model.ErrorCode.GoogleJwtBearerTokenInvalid:
          setErrorMessage(
            'Your account could not be validated by Google.  Please check that your Google account is valid and try again.'
          )
          break

        case Model.ErrorCode.GoogleJwtNonceInvalid:
          setErrorMessage(
            'Your account could not be validated by Google, the nonce is invalid.  Please check that your Google account is valid and try again.'
          )
          break

        default:
          setErrorMessage(error.toString())
      }
    }
  }

  return (
    <UnauthenticatedLayout
      id="signin"
      title="Welcome!"
      message="Please provide your user credentials in order to sign in."
      errorMessage={errorMessage}>
      <div
        className={`muted password-reset ${isPasswordResetLinkVisible ? '' : 'hidden'}`}>
        Forgot your password?{' '}
        <Link className="simple-link" to="/forgotpassword">
          Click here to request a password reset!
        </Link>
      </div>
      <form
        className={errorMessage.length > 0 ? 'form-error' : ''}
        onSubmit={handleSubmit}>
        <div className="mb-3">
          <TextInput
            required={true}
            type="text"
            label="Email address"
            name="email-address"
            value={emailAddress}
            onChange={(value: string) => setEmailAddress(value)}></TextInput>
        </div>
        <div className="mb-3">
          <TextInput
            required={true}
            type="password"
            label="Password"
            name="password"
            value={password}
            onChange={(value: string) => setPassword(value)}></TextInput>
        </div>
        <div className="form-check">
          <input id="rememberPassword" type="checkbox" className="form-check-input" />
          <label htmlFor="rememberPassword" className="form-check-label">
            Remember password
          </label>
        </div>
        <div className="d-grid gap-2 mt-2">
          <button
            disabled={!isSubmitButtonEnabled}
            type="submit"
            className="styled-button">
            Sign in
          </button>
        </div>
        <div className="or-block">
          <span>OR</span>
        </div>
        <div id="social-login-buttons">
          <div id="google-login-button">
            <GoogleLogin
              nonce={nonce}
              onSuccess={handleGoogleSubmit}
              onError={handleGoogleError}
            />
            <div id="google-login-override" className="styled-button">
              <span>Sign in using Google</span>
            </div>
          </div>
        </div>
        <div className="not-registered text-muted">
          Interested in learning more?{' '}
          <Link className="simple-link" to="/register">
            Click here to get started!
          </Link>
        </div>
      </form>
    </UnauthenticatedLayout>
  )
}

export default SignIn

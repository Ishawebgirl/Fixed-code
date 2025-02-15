import TextInput from 'components/TextInput'
import { useEffect, useMemo, useRef, useState } from 'react'
import { RegexPattern } from 'helpers/constants'
import { UnauthenticatedLayout } from 'layouts/UnauthenticatedLayout'
import { v4 } from 'uuid'
import { Link, useNavigate } from 'react-router-dom'
import { GoogleLogin } from '@react-oauth/google'
import * as Model from 'api/model'

const Register = () => {
  const customerApi = useMemo(
    () => new Model.CustomerClient(process.env.REACT_APP_API_URL),
    []
  )

  const [isSubmitButtonEnabled, setIsSubmitButtonEnabled] = useState(false)
  const [isPasswordAllowed] = useState(false)
  const [firstName, setFirstName] = useState('')
  const [lastName, setLastName] = useState('')
  const [companyName, setCompanyName] = useState('')
  const [address, setAddress] = useState('')
  const [address2, setAddress2] = useState('')
  const [city, setCity] = useState('')
  const [state, setState] = useState('')
  const [postalCode, setPostalCode] = useState('')
  const [emailAddress, setEmailAddress] = useState('')
  const [password, setPassword] = useState('')
  const [password2, setPassword2] = useState('')
  const [telephone, setTelephone] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [nonce] = useState(v4())

  const navigate = useRef(useNavigate())

  useEffect(() => {
    // reenable this if you want investigators to create passwords upon registration, rather than after email confirmation
    // const params = new URLSearchParams(window.location.search);
    // let passwordParam = params.get("password");
    // setIsPasswordAllowed(passwordParam === "true");
  }, [])

  useEffect(() => {
    setIsSubmitButtonEnabled(
      firstName.length > 0 &&
        lastName.length > 0 &&
        emailAddress.length > 0 &&
        (!isPasswordAllowed || (password.length > 0 && password2.length > 0)) &&
        new RegExp(RegexPattern.emailAddress).test(emailAddress) &&
        new RegExp(RegexPattern.telephone).test(telephone)
    )
  }, [
    isPasswordAllowed,
    firstName,
    lastName,
    companyName,
    address,
    address2,
    city,
    state,
    postalCode,
    telephone,
    emailAddress,
    password,
    password2,
  ])

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()

    if (!validate()) {
      return
    }

    await attemptRegister(async () => {
      const request = Model.CustomerRegistration.fromJS({
        name: companyName,
        firstName: firstName,
        lastName: lastName,
        address: address,
        address2: address2,
        city: city,
        state: state,
        postalCode: postalCode,
        telephone: telephone,
        emailAddress: emailAddress,
        password: password,
      })

      await customerApi.register(request)
    })
  }

  const handleGoogleSubmit = async (credentialResponse: any) => {
    if (credentialResponse == null || credentialResponse.credential == null) {
      setErrorMessage(
        'An error occurred while attempting to sign in via Google: the remote system failed to return a valid credential token.'
      )
      return
    }

    await attemptRegister(async () => {
      await customerApi.register(credentialResponse.credential)
    })
  }

  const handleGoogleError = async () => {
    setErrorMessage('An error occurred while attempting to sign in via Google.')
  }

  const validate = () => {
    if (
      firstName.length === 0 ||
      lastName.length === 0 ||
      companyName.length === 0 ||
      address.length === 0 ||
      city.length === 0 ||
      state.length === 0 ||
      postalCode.length === 0 ||
      telephone.length === 0 ||
      emailAddress.length === 0
    ) {
      setErrorMessage(' ')
      return false
    }

    if (!new RegExp(RegexPattern.emailAddress).test(emailAddress)) {
      setErrorMessage('The email address you provided is not valid.')
      return false
    }

    if (isPasswordAllowed) {
      if (password.length === 0 || password2.length === 0) {
        setErrorMessage(' ')
        return false
      }

      if (password !== password2) {
        setErrorMessage('The passwords do not match.')
        return false
      }

      if (!new RegExp(RegexPattern.password).test(password)) {
        setErrorMessage(
          'The password does not meet the minimum complexity requirements.  Please make sure that your password is at least 8 characters and includes a lowercase letter, an uppercase letter, a number, and a symbol.'
        )
        return false
      }
    }

    return true
  }

  const attemptRegister = async (registerFunction: () => Promise<void>) => {
    await registerFunction()
      .then((result) => {
        navigate.current('/thankyou')
      })
      .catch((error) => {
        if (error instanceof TypeError && error.message === 'Failed to fetch') {
          setErrorMessage(
            'The request could not be completed, the backend API may not be configured correctly.'
          )
          return
        }

        const apiError = JSON.parse(error.response)

        switch (apiError?.errorCodeName) {
          case Model.ErrorCode.AccountEmailAddressInvalid:
            setErrorMessage('The email address you provided is not valid.')
            break

          case Model.ErrorCode.AccountPasswordDoesNotMeetMinimumComplexity:
            setErrorMessage(
              'The password does not meet the minimum complexity requirements.  Please make sure that your password is at least 8 characters and includes a lowercase letter, an uppercase letter, a number, and a symbol.'
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
      })
  }

  return (
    <UnauthenticatedLayout
      id="register"
      title="Learn More"
      message="Please enter your company contact information below."
      errorMessage={errorMessage}
      reversed={true}>
      <form
        className={errorMessage.length > 0 ? 'form-error' : ''}
        onSubmit={handleSubmit}>
        <div className="row">
          <div className="col-lg-6">
            <TextInput
              type="text"
              label="First name"
              name="first-name"
              value={firstName}
              required={true}
              onChange={(value: string) => setFirstName(value)}></TextInput>
          </div>
          <div className="col-lg-6">
            <TextInput
              type="text"
              label="Last name"
              name="last-name"
              value={lastName}
              required={true}
              onChange={(value: string) => setLastName(value)}></TextInput>
          </div>
        </div>
        <div className="mb-3">
          <TextInput
            type="text"
            label="Company name"
            name="company-name"
            value={companyName}
            required={true}
            onChange={(value: string) => setCompanyName(value)}></TextInput>
        </div>
        <div className="mb-3">
          <TextInput
            type="text"
            label="Address"
            name="address"
            value={address}
            required={true}
            onChange={(value: string) => setAddress(value)}></TextInput>
        </div>
        <div className="mb-3">
          <TextInput
            type="text"
            label="Apartment, suite, etc."
            name="address2"
            value={address2}
            required={false}
            onChange={(value: string) => setAddress2(value)}></TextInput>
        </div>
        <div className="mb-3">
          <TextInput
            type="text"
            label="City"
            name="city"
            value={city}
            required={true}
            onChange={(value: string) => setCity(value)}></TextInput>
        </div>
        <div className="row">
          <div className="col-lg-6">
            <TextInput
              type="state"
              label="State"
              name="state"
              value={state}
              required={true}
              onChange={(value: string) => setState(value)}></TextInput>
          </div>
          <div className="col-lg-6">
            <TextInput
              type="text"
              label="Postal Code"
              name="postalCode"
              regex={RegexPattern.postalCode}
              value={postalCode}
              required={true}
              onChange={(value: string) => setPostalCode(value)}></TextInput>
          </div>
        </div>
        <div className="mb-3">
          <TextInput
            type="text"
            label="Email address"
            name="email-address"
            value={emailAddress}
            required={true}
            onChange={(value: string) => setEmailAddress(value)}></TextInput>
        </div>
        <div className="mb-3">
          <TextInput
            type="telephone"
            label="Telephone"
            name="telephone"
            value={telephone}
            required={true}
            onChange={(value: string) => setTelephone(value)}></TextInput>
        </div>
        {isPasswordAllowed && (
          <>
            <div className="mb-3">
              <TextInput
                type="password"
                label="Password"
                name="password"
                value={password}
                required={true}
                onChange={(value: string) => setPassword(value)}></TextInput>
            </div>
            <div className="mb-3">
              <TextInput
                type="password"
                label="Retype password"
                name="password2"
                value={password2}
                required={true}
                onChange={(value: string) => setPassword2(value)}></TextInput>
            </div>
          </>
        )}
        <div className="d-grid gap-2 mt-2">
          <button
            disabled={!isSubmitButtonEnabled}
            type="submit"
            className="styled-button">
            Submit
          </button>
        </div>
        <div className="or-block" style={{ display: 'none' }}>
          <span>OR</span>
        </div>
        <div id="social-login-buttons" style={{ display: 'none' }}>
          <div id="google-login-button">
            <GoogleLogin
              nonce={nonce}
              onSuccess={(credentialResponse) => {
                handleGoogleSubmit(credentialResponse)
              }}
              onError={handleGoogleError}></GoogleLogin>
            <div id="google-login-override" className="styled-button">
              <span>Register using Google</span>
            </div>
          </div>
        </div>
        <div className="not-registered text-muted">
          <Link className="simple-link" to="/signin">
            Return to sign-in page
          </Link>
        </div>
      </form>
    </UnauthenticatedLayout>
  )
}

export default Register

import TextInput from 'components/TextInput'
import { useEffect, useMemo, useRef, useState } from 'react'
import { UnauthenticatedLayout } from 'layouts/UnauthenticatedLayout'
import { RegexPattern } from 'helpers/constants'
import { Link, useNavigate } from 'react-router-dom'
import * as Model from 'api/model'

const ForgotPassword = () => {
  const accountApi = useMemo(
    () => new Model.AccountClient(process.env.REACT_APP_API_URL),
    []
  )

  const [emailAddress, setEmailAddress] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [isSubmitButtonEnabled, setIsSubmitButtonEnabled] = useState(false)

  const navigate = useRef(useNavigate())

  useEffect(() => {
    setIsSubmitButtonEnabled(emailAddress.length > 0)
  }, [emailAddress])

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()

    if (!validate()) {
      return
    }

    await attemptSignin()
  }

  const validate = () => {
    if (emailAddress.length === 0) {
      setErrorMessage(' ')
      return false
    }

    if (!new RegExp(RegexPattern.emailAddress).test(emailAddress)) {
      setErrorMessage('The email address you provided is not valid.')
      return false
    }

    return true
  }

  const attemptSignin = async () => {
    await accountApi
      .requestResetPassword(new Model.PasswordResetRequest({ emailAddress }))
      .then((result) => {
        navigate.current('/thankyou')
      })
      .catch((error) => {
        const apiError = JSON.parse(error.response)

        if (apiError.message) {
          setErrorMessage(apiError.message)
        } else {
          setErrorMessage(error)
        }
      })
  }

  return (
    <UnauthenticatedLayout
      id="forgotpassword"
      title="Forgot Password"
      message="Please enter your email address and click the Reset button.  We'll send you an email with instructions for resetting your password."
      errorMessage={errorMessage}
      reversed={true}>
      <form
        className={errorMessage.length > 0 ? 'form-error' : ''}
        onSubmit={handleSubmit}>
        <div className="mb-3">
          <TextInput
            type="text"
            required={true}
            label="Email address"
            name="email-address"
            value={emailAddress}
            onChange={(value: string) => setEmailAddress(value)}></TextInput>
        </div>
        <div className="d-grid gap-2 mt-2">
          <button
            disabled={!isSubmitButtonEnabled}
            type="submit"
            className="styled-button">
            Reset
          </button>
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

export default ForgotPassword

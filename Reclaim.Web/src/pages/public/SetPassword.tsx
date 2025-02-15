import TextInput from 'components/TextInput'
import { useEffect, useMemo, useRef, useState } from 'react'
import { RegexPattern } from 'helpers/constants'
import { UnauthenticatedLayout } from 'layouts/UnauthenticatedLayout'
import * as Model from 'api/model'
import { useNavigate } from 'react-router-dom'

const SetPassword = () => {
  const accountApi = useMemo(
    () => new Model.AccountClient(process.env.REACT_APP_API_URL),
    []
  )

  const [title, setTitle] = useState('')
  const [message, setMessage] = useState('')
  const [emailAddress, setEmailAddress] = useState('')
  const [token, setToken] = useState('')
  const [password, setPassword] = useState('')
  const [password2, setPassword2] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [isSubmitButtonEnabled, setIsSubmitButtonEnabled] = useState(false)
  const [isSuccessful, setIsSuccessful] = useState(false)

  const navigate = useRef(useNavigate())

  useEffect(() => {
    setIsSubmitButtonEnabled(password.length > 0 && password2.length > 0)
  }, [password, password2])

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    let paramToken = params.get('token')
    let paramEmailAddress = params.get('emailAddress')
    let paramsIsAccountConfirmed = params.get('isAccountConfirmed') !== 'false'

    if (paramToken === null || paramEmailAddress === null) {
      setErrorMessage(
        'The link does not contain the required information in order to confirm your account.  Please try copying and pasting the link directly from the email you received.'
      )
    }

    setEmailAddress(paramEmailAddress!)
    setToken(paramToken!)

    if (paramsIsAccountConfirmed) {
      setTitle('Create New Password')
      setMessage('Please provide a new password in the fields below.')
    } else {
      setTitle('Account Confirmation')
      setMessage("Welcome!  Before we get started, let's have you create a new password.")
    }
  }, [])

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()

    if (!validate()) {
      return
    }

    await apiSetPassword()
  }

  const validate = () => {
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

    return true
  }

  const apiSetPassword = async () => {
    const request = Model.PasswordReset.fromJS({
      emailAddress: emailAddress,
      token: token,
      newPassword: password,
    })

    await accountApi
      .resetPassword(request)
      .then((result) => {
        setTitle('Success!')
        setMessage(
          "You've created a new password and are ready to go.  Please log in using these credentials on the following screen!"
        )
        setIsSuccessful(true)

        setTimeout(() => {
          navigate.current('/signin?emailAddress=' + encodeURIComponent(emailAddress))
        }, 2000)
      })
      .catch((error) => {
        const apiError = JSON.parse(error.response)

        switch (apiError?.errorCodeName) {
          case Model.ErrorCode.AccountMagicUrlTokenExpired:
            setErrorMessage(
              "Your account confirmation link has expired.  Please <a href='/forgotpassword'>request a new password</a> to generate a new link."
            )
            break

          case Model.ErrorCode.AccountPasswordDoesNotMeetMinimumComplexity:
            setErrorMessage(
              'The password does not meet the minimum complexity requirements.  Please choose a password with at least 8 characters, including a lowercase letter, an uppercase letter, a number, and a symbol.'
            )
            break

          case Model.ErrorCode.AccountPasswordUsedPreviously:
            setErrorMessage(
              'The password has been used before by this account.  Please choose a unique password.'
            )
            break

          default:
            setErrorMessage(error?.response?.data?.message ?? JSON.stringify(error))
            break
        }
      })
  }

  return (
    <UnauthenticatedLayout
      id="setpassword"
      title={title}
      message={message}
      errorMessage={errorMessage}
      reversed={true}>
      <form
        className={
          (errorMessage.length > 0 ? 'form-error' : '') + (isSuccessful ? 'hidden' : '')
        }
        onSubmit={handleSubmit}>
        <div className="mb-3">
          <TextInput
            type="password"
            label="Password"
            name="password"
            required={true}
            value={password}
            onChange={(value: string) => setPassword(value)}></TextInput>
        </div>
        <div className="mb-3">
          <TextInput
            type="password"
            label="Retype password"
            name="password2"
            required={true}
            value={password2}
            onChange={(value: string) => setPassword2(value)}></TextInput>
        </div>
        <div className="d-grid gap-2 mt-2">
          <button
            disabled={!isSubmitButtonEnabled}
            type="submit"
            className="styled-button">
            Submit
          </button>
        </div>
      </form>
    </UnauthenticatedLayout>
  )
}

export default SetPassword

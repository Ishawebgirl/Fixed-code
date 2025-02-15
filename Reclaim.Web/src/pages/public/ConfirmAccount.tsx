import { useEffect, useMemo, useState } from 'react'
import { UnauthenticatedLayout } from 'layouts/UnauthenticatedLayout'
import * as Model from 'api/model'
import { Link } from 'react-router-dom'

const ConfirmAccount = () => {
  const accountApi = useMemo(
    () => new Model.AccountClient(process.env.REACT_APP_API_URL),
    []
  )

  const [emailAddress, setEmailAddress] = useState('')
  const [token, setToken] = useState('')
  const [isPasswordResetRequired] = useState(false)
  const [title, setTitle] = useState('Confirming Account')
  const [message, setMessage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [isConfirmed, setIsConfirmed] = useState(false)

  useEffect(() => {
    const apiConfirmAccount = async (
      emailAddress: string,
      token: string
    ): Promise<void> => {
      await accountApi
        .confirm(
          new Model.AccountConfirmation({ emailAddress: emailAddress, token: token })
        )
        .then(() => {
          // note, even if the email or token are wrong, this will display.
          // I don't want to have an un-authenticated method that allows bots
          // to browse for valid accounts
          setTitle('Account Confirmed!')
          setMessage('Please click the button below to sign in and get started.')
          setIsConfirmed(true)
        })
        .catch((error) => {
          const apiError = JSON.parse(error.response)

          switch (apiError?.errorCodeName) {
            case Model.ErrorCode.AccountAlreadyConfirmed:
              setMessage(
                'Your account has already been confirmed.  Click the button below to sign in and get started!'
              )
              setIsConfirmed(true)
              return

            case Model.ErrorCode.AccountMagicUrlTokenExpired:
              setErrorMessage(
                "Your account confirmation link has expired.  Please <a href='/forgotpassword'>request a new password</a> to generate a new link."
              )
              break

            default:
              setErrorMessage(error?.response?.data?.message ?? JSON.stringify(error))
              break
          }

          setTitle('Account Confirmation Failed')
        })
    }

    ;(async () => {
      const params = new URLSearchParams(window.location.search)
      const paramToken = params.get('token')
      const paramEmailAddress = params.get('emailAddress')

      if (paramToken === null || paramEmailAddress === null) {
        setErrorMessage(
          'The link does not contain the required information in order to confirm your account.  Please try copying and pasting the link directly from the email you received.'
        )
        return
      }

      setEmailAddress(paramEmailAddress!)
      setToken(paramToken!)

      await apiConfirmAccount(paramEmailAddress!, paramToken!)
    })()
  }, [accountApi])

  return (
    <UnauthenticatedLayout
      id="confirmaccount"
      title={title}
      message={message}
      errorMessage={errorMessage}>
      {isConfirmed && (
        <form>
          <div>
            <div>
              {isPasswordResetRequired ? (
                <Link
                  to={`/setpassword?emailAddress=${encodeURIComponent(emailAddress)}&token=${token}&isAccountConfirmed=false`}
                  className="styled-button">
                  Create new password
                </Link>
              ) : (
                <Link
                  to={`/signin?emailAddress=${encodeURIComponent(emailAddress)}`}
                  className="styled-button">
                  Sign in
                </Link>
              )}
            </div>
          </div>
        </form>
      )}
    </UnauthenticatedLayout>
  )
}

export default ConfirmAccount

import { UnauthenticatedLayout } from 'layouts/UnauthenticatedLayout'

const ThankYou = () => {
  return (
    <UnauthenticatedLayout
      id="thankyou"
      title="Thank you!"
      message="We just sent you an email with a confirmation link.  You'll need to click that link before you can sign in."
      errorMessage=""
      children={null}
    />
  )
}

export default ThankYou

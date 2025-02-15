import { Identity } from 'api/identity'
import { CookieName } from 'helpers/constants'
import Cookies from 'js-cookie'

const useIdentity = () => {
  const identityCookieName = CookieName.identity
  const identityCookie = Cookies.get(identityCookieName)

  if (identityCookie !== undefined) {
    const identity = Identity.parse(identityCookie)
    return identity
  }

  return null
}

export default useIdentity

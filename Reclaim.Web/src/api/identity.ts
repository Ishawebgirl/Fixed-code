import { Role } from './model'

export class Identity {
  public role: string
  public emailAddress: string
  public avatarUrl: string
  public niceName: string
  public nonce: string | null
  public expiration: string

  constructor(
    _role: string,
    _emailAddress: string,
    _avatarUrl: string,
    _niceName: string,
    _nonce: string | null,
    _expiration: string
  ) {
    this.role = _role
    this.emailAddress = _emailAddress
    this.avatarUrl = _avatarUrl
    this.niceName = _niceName
    this.nonce = _nonce
    this.expiration = _expiration
  }

  hasRole(role: Role) {
    return this.role === role.toString()
  }

  isAdministrator() {
    return this.role === Role.Administrator.toString()
  }

  isCustomer() {
    return this.role === Role.Customer.toString()
  }

  isInvestigator() {
    return this.role === Role.Investigator.toString()
  }

  static parse(json: string) {
    try {
      const data = JSON.parse(json)
      return new Identity(
        data.role,
        data.emailAddress,
        data.avatarUrl,
        data.niceName,
        data.nonce,
        data.expiration
      )
    } catch (e) {
      return null
    }
  }
}

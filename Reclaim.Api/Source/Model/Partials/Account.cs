namespace Reclaim.Api.Model;

public enum AccountStatus
{
    Normal = 1,
    EmailAddressNotConfirmed = 2,
    PasswordExpired = 3,
    LockedOut = 4,
    Tombstoned = 5
}

public partial class Account
{
    public AccountStatus Status
    {
        get
        {
            if (this.TombstonedTimestamp != null)
                return AccountStatus.Tombstoned;
            else if (this.LockedOutTimestamp != null && this.LockedOutTimestamp.Value.AddSeconds(Setting.AccountLockedOutTimeout) > DateTime.UtcNow)
                return AccountStatus.LockedOut;
            else if (this.PasswordExpiredTimestamp != null)
                return AccountStatus.PasswordExpired;
            else if (this.EmailAddressConfirmedTimestamp == null)
                return AccountStatus.EmailAddressNotConfirmed;
            else
                return AccountStatus.Normal;
        }
    }
}
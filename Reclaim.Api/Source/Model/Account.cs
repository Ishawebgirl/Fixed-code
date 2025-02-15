using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class Account : Base
{
    public int ID { get; set; }

    public Role Role { get; set; }

    public IdentityProvider IdentityProvider { get; set; }

    public string EmailAddress { get; set; } = null!;

    public string PasswordHash { get; set; } = null!;

    public string PasswordSalt { get; set; } = null!;

    public string? AvatarUrl { get; set; }

    public string NiceName { get; set; } = null!;

    public Guid? MagicUrlToken { get; set; }

    public DateTime? MagicUrlValidUntil { get; set; }

    public DateTime? AuthenticatedTimestamp { get; set; }

    public DateTime? SessionAuthenticatedTimestamp { get; set; }

    public DateTime? LastActiveTimestamp { get; set; }

    public int FailedAuthenticationCount { get; set; }

    public int BouncedEmailCount { get; set; }

    public DateTime? BouncedEmailTimestamp { get; set; }

    public DateTime? PasswordExpiredTimestamp { get; set; }

    public DateTime? PasswordChangedTimestamp { get; set; }

    public DateTime? EmailAddressConfirmedTimestamp { get; set; }

    public DateTime? TombstonedTimestamp { get; set; }

    public DateTime? LockedOutTimestamp { get; set; }

    public Guid UniqueID { get; set; }



    public virtual ICollection<AccountAuthentication> Authentications { get; set; } = new List<AccountAuthentication>();

    public virtual ICollection<AccountPassword> Passwords { get; set; } = new List<AccountPassword>();

    public virtual ICollection<Administrator> Administrators { get; set; } = new List<Administrator>();

    public virtual ICollection<Chat> Chats { get; set; } = new List<Chat>();

    public virtual ICollection<Claim> Claims { get; set; } = new List<Claim>();

    public virtual Customer? Customer { get; set; }

    public virtual ICollection<Document> Documents { get; set; } = new List<Document>();

    public virtual ICollection<Email> Emails { get; set; } = new List<Email>();

    public virtual Investigator? Investigator { get; set; }

    public virtual ICollection<LogEntry> LogEntries { get; set; } = new List<LogEntry>();
}

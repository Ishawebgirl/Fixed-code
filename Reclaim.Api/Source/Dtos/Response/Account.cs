using System.ComponentModel.DataAnnotations;
using System.Diagnostics.CodeAnalysis;

namespace Reclaim.Api.Dtos;

public record Account : Base
{
    /// <example>cc297118-87a5-40f1-9639-4c24fbc647e6</example>
    [Required]
    public Guid UniqueID { get; set; }

    /// <example>Customer</example>
    [Required]
    public Model.Role Role { get; set; }

    /// <example>Local</example>
    [Required]
    public Model.IdentityProvider IdentityProvider { get; set; }

    /// <example>customer@test.com</example>
    [Required]
    public string EmailAddress { get; set; }

    /// <example>https://reclaimblob.blob.core.windows.net/avatar/6075DF53-D4CC-4714-B8AE-6ED6D8E28B8B.jpg</example>
    public string? AvatarUrl { get; set; }

    /// <example>Rex McCrie</example>
    public string? NiceName { get; set; }

    public DateTime? AuthenticatedTimestamp { get; set; }
    public DateTime? SessionAuthenticatedTimestamp { get; set; }
    public DateTime? LastActiveTimestamp { get; set; }
    public DateTime? EmailAddressConfirmedTimestamp { get; set; }

    [SetsRequiredMembers]
    public Account(Model.Account model)
    {
        UniqueID = model.UniqueID;
        Role = model.Role;
        IdentityProvider = model.IdentityProvider;
        EmailAddress = model.EmailAddress;
        NiceName = model.NiceName;
        AvatarUrl = model.AvatarUrl;
        AuthenticatedTimestamp = model.AuthenticatedTimestamp;
        SessionAuthenticatedTimestamp = model.SessionAuthenticatedTimestamp;
        LastActiveTimestamp = model.LastActiveTimestamp;
        EmailAddressConfirmedTimestamp = model.EmailAddressConfirmedTimestamp;
    }
}

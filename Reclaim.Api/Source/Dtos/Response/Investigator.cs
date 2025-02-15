using Reclaim.Api.Model;
using System.ComponentModel.DataAnnotations;

namespace Reclaim.Api.Dtos;

public record Investigator
{
    /// <example>1ee83aeb-6bfb-453a-9107-6b3d85740709</example>
    [Required]
    public Guid UniqueID { get; set; }

    /// <example>Active</example>
    [Required]
    public InvestigatorStatus Status { get; set; }

    /// <example>Hal</example>
    [Required]
    public string FirstName { get; set; }

    /// <example>Nichols</example>
    [Required]
    public string LastName { get; set; }

    /// <example>957 Kennedy Place</example>
    [Required]
    public string Address { get; set; }

    /// <example>null</example>
    public string? Address2 { get; set; }

    /// <example>Kansas City</example>
    [Required]
    public string City { get; set; }

    /// <example>MO</example>
    [Required]
    public string State { get; set; }

    /// <example>64030</example>
    [Required]
    public string PostalCode { get; set; }

    /// <example>+1 748-636-8718</example>
    [Required]
    public string Telephone { get; set; }

    /// <example>hnichols@test.com</example>
    [Required]
    public string EmailAddress { get; set; }

    /// <example>https://reclaimblob.blob.core.windows.net/avatar/6075DF53-D4CC-4714-B8AE-6ED6D8E28B8B.jpg</example>
    public string AvatarUrl { get; set; }

    public DateTime? LastActiveTimestamp { get; set; }

    public Investigator(Model.Investigator model)
    {
        UniqueID = model.UniqueID;
        Status = model.Status;
        FirstName = model.FirstName;
        LastName = model.LastName;
        Address = model.Address;
        Address2 = model.Address2;
        City = model.City;
        State = model.State.Code;
        PostalCode = model.PostalCode;
        Telephone = model.Telephone;
        EmailAddress = model.Account.EmailAddress;
        AvatarUrl = model.Account.AvatarUrl;
        LastActiveTimestamp = model.Account.LastActiveTimestamp;
    }
}
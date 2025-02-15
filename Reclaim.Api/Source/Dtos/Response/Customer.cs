using Reclaim.Api.Model;
using System.ComponentModel.DataAnnotations;

namespace Reclaim.Api.Dtos;

public record Customer
{
    /// <example>60f80c13-7af4-4c3a-92e2-399549b2be3b</example>
    [Required]
    public Guid UniqueID { get; set; }

    /// <example>Active</example>
    [Required]
    public CustomerStatus Status { get; set; }

    /// <example>Harpeth Consulting</example>
    [Required]
    public string Name { get; set; }

    /// <example>Alison</example>
    [RequiredString(100)]
    public string FirstName { get; set; }

    /// <example>Proud</example>
    [RequiredString(100)]
    public string LastName { get; set; }

    /// <example>HARPETH</example>
    [Required]
    public string Code { get; set; }

    /// <example>1520 Horton Ave</example>
    [Required]
    public string Address { get; set; }

    /// <example>Suite 403</example>
    public string? Address2 { get; set; }

    /// <example>Nashville</example>
    [Required]
    public string City { get; set; }

    /// <example>TN</example>
    [Required]
    public string State { get; set; }

    /// <example>37212</example>
    [Required]
    public string PostalCode { get; set; }

    /// <example>+1 615-284-1128</example>
    [Required]
    public string Telephone { get; set; }

    /// <example>customer@test.com</example>
    [Required]
    public string EmailAddress { get; set; }

    /// <example>https://reclaimblob.blob.core.windows.net/avatar/6075DF53-D4CC-4714-B8AE-6ED6D8E28B8B.jpg</example>
    public string? AvatarUrl { get; set; }

    public DateTime? LastActiveTimestamp { get; set; }

    public Customer(Model.Customer model)
    {
        UniqueID = model.UniqueID;
        Status = model.Status;
        Name = model.Name;
        Code = model.Code;
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
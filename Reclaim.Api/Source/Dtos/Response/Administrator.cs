using Reclaim.Api.Model;
using System.ComponentModel.DataAnnotations;

namespace Reclaim.Api.Dtos;

public record Administrator
{
    /// <example>60f80c13-7af4-4c3a-92e2-399549b2be3b</example>
    [Required]
    public Guid UniqueID { get; set; }

    /// <example>Caleb</example>
    [RequiredString(100)]
    public string FirstName { get; set; }

    /// <example>Unruh</example>
    [RequiredString(100)]
    public string LastName { get; set; }
    
    /// <example>customer@test.com</example>
    [Required]
    public string EmailAddress { get; set; }

    /// <example>https://reclaimblob.blob.core.windows.net/avatar/6075DF53-D4CC-4714-B8AE-6ED6D8E28B8B.jpg</example>
    public string? AvatarUrl { get; set; }

    public DateTime? LastActiveTimestamp { get; set; }
    
    public Administrator(Model.Administrator model)
    {
        UniqueID = model.UniqueID;
        FirstName = model.FirstName;
        LastName = model.LastName;
        EmailAddress = model.Account.EmailAddress;
        AvatarUrl = model.Account.AvatarUrl;
        LastActiveTimestamp = model.Account.LastActiveTimestamp;
    }
}
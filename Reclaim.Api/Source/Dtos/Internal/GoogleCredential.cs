namespace Reclaim.Api.Dtos;

public record GoogleCredential : Base
{
    /// <example>hnichols@test.com</example>
    public string EmailAddress { get; set; }

    /// <example>Hal</example>
    public string FirstName { get; set; }

    /// <example>Nichols</example>
    public string LastName { get; set; }

    public DateTime Expiration { get; set; }
}
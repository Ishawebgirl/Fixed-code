namespace Reclaim.Api.Dtos;

public record AccountConfirmation : Base
{
    /// <example>customer@test.com</example>
    [RequiredString(250)]
    [EmailAddress]
    public string EmailAddress { get; set; }

    /// <example>b71b93a0-dfe5-4dfe-8bf4-21b3c8702fcb</example> 
    [RequiredString]
    public string Token { get; set; }
}
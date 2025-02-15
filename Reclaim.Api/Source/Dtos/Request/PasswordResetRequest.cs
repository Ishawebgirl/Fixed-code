namespace Reclaim.Api.Dtos;

public class PasswordResetRequest
{
    /// <example>customer@test.com</example>
    [RequiredString(250)]
    [EmailAddress]
    public string EmailAddress { get; set; }
}
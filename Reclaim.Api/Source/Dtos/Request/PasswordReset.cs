namespace Reclaim.Api.Dtos;

public record PasswordReset : Base
{
    /// <example>customer@test.com</example>
    [RequiredString(250)]
    [EmailAddress]
    public string EmailAddress { get; set; }

    /// <example>7d6Z!9S1</example>
    [RequiredString]
    public string NewPassword { get; set; }

    /// <example>0dc119fb-d663-44d6-a7c1-bb4549023de4</example>
    [RequiredString]
    public string Token { get; set; }
}
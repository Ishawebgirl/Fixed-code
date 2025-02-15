namespace Reclaim.Api.Dtos;

public record AccountAuthentication : Base
{
    /// <example>customer@test.com</example>
    [RequiredString(250)]
    public string EmailAddress { get; set; }

    /// <example>kd8%8QWx</example>
    [RequiredString]
    public string Password { get; set; }
}

public record GoogleAccountAuthentication : Base
{
    /// <example>customer@test.com</example>
    [RequiredString(250)]
    [EmailAddress]
    public string EmailAddress { get; set; }

    /// <example>eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqd ...</example>
    [RequiredString]
    public string GoogleJwt { get; set; }
}

public record AccountAuthenticationRefresh : Base
{
    /// <example>customer@test.com</example>
    [RequiredString(250)]
    [EmailAddress]
    public string EmailAddress { get; set; }

    /// <example>bf973a98-d41d-4a7c-a434-bb14c5e55b19e</example>
    [RequiredString]
    public string RefreshToken { get; set; }
}
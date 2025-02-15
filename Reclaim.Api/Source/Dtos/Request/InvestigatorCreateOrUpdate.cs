namespace Reclaim.Api.Dtos;

public record InvestigatorCreateOrUpdate : Base
{
    /// <example>Hal</example>
    [RequiredString(100)]
    public string FirstName { get; set; }

    /// <example>Nichols</example>
    [RequiredString(100)]
    public string LastName { get; set; }

    /// <example>957 Kennedy Place</example>
    [RequiredString(100)]
    public string Address { get; set; }

    /// <example>null</example>
    [NonRequiredString(100)]
    public string? Address2 { get; set; }

    /// <example>Kansas City</example>
    [RequiredString(100)]
    public string City { get; set; }

    /// <example>MO</example>
    [RequiredString(2)]
    public string State { get; set; }

    /// <example>64030</example>
    [RequiredString(10)]
    [PostalCode]
    public string PostalCode { get; set; }


    /// <example>+1 748-636-8718</example>
    [RequiredString(15)]
    [Telephone]
    public string Telephone { get; set; }

    /// <example>hnichols@test.com</example>
    [RequiredString(250)]
    [EmailAddress]
    public string EmailAddress { get; set; }
}
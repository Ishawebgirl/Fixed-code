namespace Reclaim.Api.Dtos;

public record CustomerCreateOrUpdate : Base
{
    /// <example>Harpeth Consulting</example>
    [RequiredString(50)]
    public string Name { get; set; }

    /// <example>HARPETH</example>
    [RequiredString(10)]
    [System.ComponentModel.DataAnnotations.RegularExpression(RegularExpression.CustomerCode)]
    public string Code { get; set; }

    /// <example>Alison</example>
    [RequiredString(100)]
    public string FirstName { get; set; }

    /// <example>Proud</example>
    [RequiredString(100)]
    public string LastName { get; set; }

    /// <example>1520 Horton Ave</example>
    [RequiredString(100)]
    public string Address { get; set; }

    /// <example>Suite 403</example>
    [NonRequiredString(100)]
    public string? Address2 { get; set; }

    /// <example>Nashville</example>
    [RequiredString(100)]
    public string City { get; set; }

    /// <example>TN</example>
    [RequiredString(2)]
    public string State { get; set; }

    /// <example>37212</example>
    [RequiredString(10)]
    [PostalCode]
    public string PostalCode { get; set; }

    /// <example>customer@test.com</example>
    [RequiredString(250)]
    [EmailAddress]
    public string EmailAddress { get; set; }

    /// <example>+1 615-284-1128</example>
    [RequiredString(15)]
    [Telephone]
    public string Telephone { get; set; }
}

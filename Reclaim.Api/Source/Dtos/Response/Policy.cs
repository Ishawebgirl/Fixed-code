using Reclaim.Api.Model;
using System.Diagnostics.CodeAnalysis;
using System.ComponentModel.DataAnnotations;

namespace Reclaim.Api.Dtos;

public record Policy
{
    /// <example>e7e2e6f1-55ae-4207-ae9f-4d2a5f52bcd1</example>
    [Required]
    public Guid UniqueID { get; set; }

    /// <example>0091489-001-HS</example>
    [Required]
    public string ExternalID { get; set; }

    public DateOnly? BindingDate { get; set; }
    public DateOnly? StartDate { get; set; }
    public DateOnly? EndDate { get; set; }

    /// <example>1000</example>
    public decimal? Deductible { get; set; }

    /// <example>2242.10</example>
    public decimal? AnnualPremium { get; set; }

    /// <example>0</example>
    public int? ClaimsInLastYear { get; set; }

    /// <example>1</example>
    public int? ClaimsInLast3Years { get; set; }

    /// <example>Timothy</example>
    [Required]
    public string FirstName { get; set; }

    /// <example>Wilcox</example>
    [Required]
    public string LastName { get; set; }

    /// <example>170 Claremont Park</example>
    [Required]
    public string Address { get; set; }

    /// <example>null</example>
    public string? Address2 { get; set; }

    /// <example>Trenton</example>
    [Required]
    public string City { get; set; }

    /// <example>NJ</example>
    [Required]
    public string State { get; set; }

    /// <example>08601</example>
    [Required]
    public string PostalCode { get; set; }

    /// <example>+1 555-618-4242</example>
    [Required]
    public string Telephone { get; set; }

    public DateOnly? DateOfBirth { get; set; }

    /// <example>3</example>
    [Required]
    public int Bedrooms { get; set; }

    /// <example>2</example>
    public int? Bathrooms { get; set; }

    /// <example>Owned</example>
    public OwnershipType? OwnershipType { get; set; }

    /// <example>House</example>
    public PropertyType? PropertyType { get; set; }

    /// <example>Membrane</example>
    public RoofType? RoofType { get; set; }

    /// <example>2005</example>
    public int? YearBuilt { get; set; }

    [Required]
    public Customer Customer { get; set; }

    [SetsRequiredMembers]
    public Policy(Model.Policy model, bool includeCustomer = true)
    {
        UniqueID = model.UniqueID;
        ExternalID = model.ExternalID;
        BindingDate = model.BindingDate;
        StartDate = model.StartDate;
        EndDate = model.EndDate;
        Deductible = model.Deductible;
        AnnualPremium = model.AnnualPremium;
        ClaimsInLastYear = model.ClaimsInLastYear;
        ClaimsInLast3Years = model.ClaimsInLast3Years;
        FirstName = model.FirstName;
        LastName = model.LastName;
        Address = model.Address;
        Address2 = model.Address2;
        City = model.City;
        State = model.State.Code;
        PostalCode = model.PostalCode;
        Telephone = model.Telephone;
        DateOfBirth = model.DateOfBirth;
        Bedrooms = model.Bedrooms;
        Bathrooms = model.Bathrooms;
        OwnershipType = model.OwnershipType;
        PropertyType = model.PropertyType;
        RoofType = model.RoofType;
        YearBuilt = model.YearBuilt;

        if (includeCustomer)
            Customer = new Customer(model.Customer);
    }
}

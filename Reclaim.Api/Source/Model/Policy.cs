using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class Policy : Base
{
    public int ID { get; set; }

    public int CustomerID { get; set; }

    public string ExternalID { get; set; } = null!;

    public DateOnly? BindingDate { get; set; }

    public DateOnly? StartDate { get; set; }

    public DateOnly? EndDate { get; set; }

    public decimal? Deductible { get; set; }

    public decimal? AnnualPremium { get; set; }

    public int? ClaimsInLastYear { get; set; }

    public int? ClaimsInLast3Years { get; set; }

    public string FirstName { get; set; } = null!;

    public string LastName { get; set; } = null!;

    public string Address { get; set; } = null!;

    public string? Address2 { get; set; }

    public string City { get; set; } = null!;

    public int StateID { get; set; }

    public string PostalCode { get; set; } = null!;

    public string Telephone { get; set; } = null!;

    public DateOnly? DateOfBirth { get; set; }

    public int Bedrooms { get; set; }

    public int? Bathrooms { get; set; }

    public OwnershipType OwnershipType { get; set; }

    public PropertyType PropertyType { get; set; }

    public RoofType RoofType { get; set; }

    public int YearBuilt { get; set; }

    public Guid UniqueID { get; set; }



    public virtual ICollection<Claim> Claims { get; set; } = new List<Claim>();

    public virtual Customer Customer { get; set; } = null!;

    public virtual State State { get; set; } = null!;
}

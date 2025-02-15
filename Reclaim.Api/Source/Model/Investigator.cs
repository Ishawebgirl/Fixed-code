using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class Investigator : Base
{
    public int ID { get; set; }

    public InvestigatorStatus Status { get; set; }

    public int AccountID { get; set; }

    public string FirstName { get; set; } = null!;

    public string LastName { get; set; } = null!;

    public string Address { get; set; } = null!;

    public string? Address2 { get; set; }

    public string City { get; set; } = null!;

    public int StateID { get; set; }

    public string PostalCode { get; set; } = null!;

    public string Telephone { get; set; } = null!;

    public Guid UniqueID { get; set; }



    public virtual Account Account { get; set; } = null!;

    public virtual ICollection<Claim> Claims { get; set; } = new List<Claim>();

    public virtual State State { get; set; } = null!;
}

using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class Customer : Base
{
    public int ID { get; set; }

    public CustomerStatus Status { get; set; }

    public int AccountID { get; set; }

    public string Code { get; set; } = null!;

    public string Name { get; set; } = null!;

    public string FirstName { get; set; } = null!;

    public string LastName { get; set; } = null!;

    public string Address { get; set; } = null!;

    public string? Address2 { get; set; }

    public string City { get; set; } = null!;

    public int StateID { get; set; }

    public string PostalCode { get; set; } = null!;

    public string? Telephone { get; set; }

    public Guid UniqueID { get; set; }



    public virtual Account Account { get; set; } = null!;

    public virtual ICollection<Policy> Policies { get; set; } = new List<Policy>();

    public virtual State State { get; set; } = null!;
}

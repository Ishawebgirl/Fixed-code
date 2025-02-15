using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class Administrator : Base
{
    public int ID { get; set; }

    public int AccountID { get; set; }

    public string FirstName { get; set; } = null!;

    public string LastName { get; set; } = null!;

    public Guid UniqueID { get; set; }



    public virtual Account Account { get; set; } = null!;
}

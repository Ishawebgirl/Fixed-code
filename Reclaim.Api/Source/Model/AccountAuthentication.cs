using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class AccountAuthentication : Base
{
    public int ID { get; set; }

    public int? AccountID { get; set; }

    public IdentityProvider IdentityProvider { get; set; }

    public bool IsSuccessful { get; set; }

    public bool IsRefresh { get; set; }

    public bool IsShadowed { get; set; }

    public string IpAddress { get; set; } = null!;

    public DateTime AuthenticatedTimestamp { get; set; }



    public virtual Account? Account { get; set; }
}

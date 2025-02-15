using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class AccountPassword : Base
{
    public int ID { get; set; }

    public int AccountID { get; set; }

    public string PasswordHash { get; set; } = null!;

    public string PasswordSalt { get; set; } = null!;

    public DateTime ArchivedTimestamp { get; set; }



    public virtual Account Account { get; set; } = null!;
}

using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class AccountAvatar : Base
{
    public int ID { get; set; }

    public int AccountID { get; set; }

    public byte[] Avatar { get; set; } = null!;



    public virtual Account Account { get; set; } = null!;
}

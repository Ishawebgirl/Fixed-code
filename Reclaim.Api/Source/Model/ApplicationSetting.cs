using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class ApplicationSetting : Base
{
    public int ID { get; set; }

    public string Name { get; set; } = null!;

    public string Value { get; set; } = null!;

    public bool? IsSecret { get; set; }


}

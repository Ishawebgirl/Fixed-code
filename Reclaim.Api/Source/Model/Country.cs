
namespace Reclaim.Api.Model;

public partial class Country
{
    public int ID { get; set; }

    public string Name { get; set; } = null!;

    public string Abbreviation { get; set; } = null!;

    public int Sequence { get; set; }

    public bool IsEnabled { get; set; }

    public DateTime CreatedTimestamp { get; set; }

    public virtual ICollection<State> States { get; set; } = new List<State>();
}

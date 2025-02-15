using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;

namespace Reclaim.Api.Dtos;

public abstract record Base
{
    public string Serialize()
    {
        var serialized = JsonConvert.SerializeObject(this, Formatting.Indented, new JsonSerializerSettings { ContractResolver = new CamelCasePropertyNamesContractResolver() });
        return serialized;
    }
}
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Dynamic;
using Xunit;

namespace Reclaim.Api.Tests;

public static class CustomAssert
{
    public static void ThatObjectHasProperty(ExpandoObject o, string key)
    {
        Assert.True(((IDictionary<string, object?>)o).ContainsKey(key));
    }

    public static void ThatObjectDoesNotHaveProperty(ExpandoObject o, string key)
    {
        Assert.False(((IDictionary<string, object?>)o).ContainsKey(key));
    }

    public static void ThatLastLogEntryIs(string errorCodeName, string messageSnippet = "")
    {
        var row = Database.GetLastLogEntry();
        
        Assert.Equal(row.Code, errorCodeName);

        if (!string.IsNullOrEmpty(messageSnippet))
            Assert.True(row.Text.IndexOf(messageSnippet) >= 0);
    }

}

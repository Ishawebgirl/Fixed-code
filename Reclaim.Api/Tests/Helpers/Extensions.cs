using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Dynamic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace Reclaim.Api.Tests;

public static class Extensions
{
    /*
    [DebuggerHidden]
    public static bool PropertyExists(dynamic obj, string name)
    {
        if (obj == null) return false;
        if (obj is IDictionary<string, object> dict)
        {
            return dict.ContainsKey(name);
        }
        return obj.GetType().GetProperty(name) != null;
    }
    */

    [DebuggerHidden]
    public static void ShouldHaveProperty(this object @this, string property)
    {
        Assert.True(((IDictionary<String, object>)@this).ContainsKey(property));
    }

    [DebuggerHidden]
    public static void ShouldNotHaveProperty(this object @this, string property)
    {
        Assert.True(((IDictionary<String, object>)@this).ContainsKey(property));
    }

    [DebuggerHidden]
    public static void ShouldContain(this string @this, string text)
    {
        Assert.True((@this).IndexOf(text) >= 0);
    }

    [DebuggerHidden]
    public static void ShouldNotContain(this string @this, string text)
    {
        Assert.True((@this).IndexOf(text) < 0);
    }

    [DebuggerHidden]
    public static void ShouldBeBetween(this int @this, int? from, int? to)
    {
        Assert.True(@this >= from && @this <= to);
    }

    [DebuggerHidden]
    public static void ShouldBeBetween(this double @this, double? from, double? to)
    {
        Assert.True(@this >= from && @this <= to);
    }

    [DebuggerHidden]
    public static void ShouldBeGreaterThan(this int @this, int? from)
    {
        Assert.True(@this > from);
    }

    [DebuggerHidden]
    public static void ShouldBeLessThan(this int @this, int? to)
    {
        Assert.True(@this < to);
    }

    [DebuggerHidden]
    public static void ShouldBeGuid(this string @this)
    {
        Guid guid;
        Assert.True(Guid.TryParse(@this, out guid));
    }

    public static string ToIsoDateTime(this DateTime dateTime)
    {
        return dateTime.ToString("yyyy-MM-ddTHH:mm:ssZ", CultureInfo.InvariantCulture);
    }

    public static IEnumerable<dynamic> AsDynamicEnumerable(this DataTable table)
    {
        return table.AsEnumerable().Select(row => new DynamicRow(row));
    }

    private sealed class DynamicRow : DynamicObject
    {
        private readonly DataRow _row;

        internal DynamicRow(DataRow row) { _row = row; }

        public override bool TryGetMember(GetMemberBinder binder, out object? result)
        {
            var retVal = _row.Table.Columns.Contains(binder.Name);
            result = retVal ? _row[binder.Name] : null;

            if (result == System.DBNull.Value)
                result = null;

            return retVal;
        }
    }

    public static ExpandoObject Initialize(this ExpandoObject expando, dynamic obj)
    {
        var expandoDic = expando as IDictionary<string, object>;
        foreach (System.Reflection.PropertyInfo fi in obj.GetType().GetProperties())
        {
            expandoDic[fi.Name] = fi.GetValue(obj, null);
        }
        return expando;
    }
}

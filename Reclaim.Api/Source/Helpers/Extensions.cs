using Microsoft.EntityFrameworkCore;
using Newtonsoft.Json;
using System.ComponentModel;
using System.Data;
using System.Data.Common;
using System.Dynamic;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace Reclaim.Api;

public static class Extensions
{
    public static string ToShortString(this Guid guid)
    {
        var base64Guid = Convert.ToBase64String(guid.ToByteArray());

        // Replace URL unfriendly characters
        base64Guid = base64Guid.Replace('+', '-').Replace('/', '_');

        // Remove the trailing ==
        return base64Guid.Substring(0, base64Guid.Length - 2);
    }

    public static Guid FromShortString(this string str)
    {
        str = str.Replace('_', '/').Replace('-', '+');
        var byteArray = Convert.FromBase64String(str + "==");
        return new Guid(byteArray);
    }

    public static bool TryParseJson<T>(this string value, out T result)
    {
        var success = true;

        var settings = new JsonSerializerSettings
        {
            Error = (sender, args) => { success = false; args.ErrorContext.Handled = true; },
            MissingMemberHandling = MissingMemberHandling.Error
        };

        result = JsonConvert.DeserializeObject<T>(value, settings);

        return success;
    }

    public static string NiceName(this Type t)
    {
        return !t.IsGenericType || t.IsGenericTypeDefinition
            ? !t.IsGenericTypeDefinition ? t.Name : t.Name.Remove(t.Name.IndexOf('`'))
            : $"{NiceName(t.GetGenericTypeDefinition())}<{string.Join(',', t.GetGenericArguments().Select(x => NiceName(x)))}>";
    }

    public static int? ToNullableInteger(this string s)
    {
        if (s == null)
            return null;

        int? result = null;

        if (!s.TryParse(out result))
            return null;
        else
            return result;
    }

    public static DateTime? ToNullableDateTime(this string s)
    {
        if (s == null)
            return null;

        DateTime? result = null;

        if (!s.TryParse(out result))
            return null;
        else
            return result;
    }

    public static bool TryParse(this string s, out int? result)
    {
        int temp;

        if (!int.TryParse(s, out temp))
        {
            result = null;
            return false;
        }
        else
        {
            result = temp;
            return true;
        }
    }

    public static bool TryParse(this string s, out DateTime? result)
    {
        DateTime temp;

        if (!DateTime.TryParse(s, out temp))
        {
            result = null;
            return false;
        }
        else
        {
            result = temp;
            return true;
        }
    }

    public static void AddParameterWithValue(this DbCommand cmd, string name, object value)
    {
        var param = cmd.CreateParameter();

        param.ParameterName = name;
        param.Value = value;
        cmd.Parameters.Add(param);
    }


    public static List<T> MapTo<T>(this DbDataReader dr)
    {
        var list = new List<T>();
        while (dr.Read())
        {
            if (typeof(T) == typeof(string))
            {
                list.Add((T)Convert.ChangeType(dr[0], typeof(T)));
            }
            else
            {
                var obj = Activator.CreateInstance<T>();

                foreach (var prop in obj.GetType().GetProperties())
                {
                    if (!Equals(dr[prop.Name], DBNull.Value))
                        prop.SetValue(obj, dr[prop.Name], null);
                }

                list.Add(obj);
            }
        }

        return list;
    }

    /*
    public static async IAsyncEnumerable<dynamic> DynamicListFromSql(this DbContext db, string sql, Dictionary<string, object> parameters)
    {
        using (var cmd = db.Database.Connection.CreateCommand())
        {
            cmd.CommandText = sql;
            if (cmd.Connection.State != ConnectionState.Open) { await cmd.Connection.OpenAsync(); }

            foreach (var p in parameters)
            {
                var dbParameter = cmd.CreateParameter();

                dbParameter.ParameterName = p.Key;
                dbParameter.Value = p.Value;
                cmd.Parameters.Add(dbParameter);
            }

            using (var dataReader = cmd.ExecuteReader())
            {
                while (await dataReader.ReadAsync())
                {
                    var row = new ExpandoObject() as IDictionary<string, object>;
                    
                    for (var fieldCount = 0; fieldCount < dataReader.FieldCount; fieldCount++)
                        row.Add(dataReader.GetName(fieldCount), dataReader[fieldCount] == DBNull.Value ? null : dataReader[fieldCount]);
                    
                    yield return row;
                }
            }
        }
    }
    */

    public static ExpandoObject Initialize(this ExpandoObject expando, dynamic obj)
    {
        var expandoDic = expando as IDictionary<string, object>;
        foreach (System.Reflection.PropertyInfo fi in obj.GetType().GetProperties())
        {
            expandoDic[fi.Name] = fi.GetValue(obj, null);
        }
        return expando;
    }

    public static string ToIsoDateTime(this DateTime dateTime)
    {
        return dateTime.ToString("yyyy-MM-ddTHH:mm:ssZ", CultureInfo.InvariantCulture);
    }

    public static string ToTitleCase(this string s)
    {
        return CultureInfo.InvariantCulture.TextInfo.ToTitleCase(s.ToLower());
    }

    public static string ToNameCase(this string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return null;

        value = value.Trim();

        if (string.Compare(value, value.ToUpper(), false) == 0)
            return value.ToTitleCase();

        if (string.Compare(value, value.ToLower(), false) == 0)
            return value.ToTitleCase();

        if (string.Compare(value.Left(1), value.Left(1).ToLower()) == 0)
            return value.ToTitleCase();

        return value;
    }

    public static string Truncate(this string value, int maxLength)
    {
        if (string.IsNullOrEmpty(value))
            return value;

        var truncated = value.Length <= maxLength ? value : value.Substring(0, maxLength);

        return truncated;
    }

    public static string GetTextBetweenTags(this string value, string tag)
    {
        var startTag = $"<{tag}>";
        var endTag = $"</{tag}";

        if (value.Contains(startTag) && value.Contains(endTag))
        {
            var index = value.IndexOf(startTag) + startTag.Length;
            return value.Substring(index, value.IndexOf(endTag) - index);
        }
        else
            return null;
    }

    public static string RemoveNonAlphabeticCharacters(this string value)
    {
        if (value == null)
            return null;
        else
            return new Regex(@"[^A-Za-z]*").Replace(value, string.Empty);
    }

    public static string RemoveNonAlphanumericCharacters(this string value)
    {
        if (value == null)
            return null;
        else
            return new Regex(@"[^A-Za-z0-9]*").Replace(value, string.Empty);
    }

    public static T Penultimate<T>(this IEnumerable<T> items)
    {
        if (items.Count() <= 1)
            return default(T);

        return (items.ElementAt(items.Count() - 2));
    }

    public static string DisplayName(this Enum value)
    {
        var field = value.GetType().GetField(value.ToString());

        var attribute = Attribute.GetCustomAttribute(field, typeof(EnumDisplayNameAttribute)) as EnumDisplayNameAttribute;

        return attribute == null ? value.ToString() : attribute.DisplayName;
    }

    private static Random rand = new Random();

    public static DateTime FirstOfMonth(this DateTime dateTime)
    {
        return new DateTime(dateTime.Year, dateTime.Month, 1);
    }

    public static string ReplaceFirst(this string text, string search, string replace)
    {
        int pos = text.IndexOf(search);

        if (pos < 0)
            return text;

        return text.Substring(0, pos) + replace + text.Substring(pos + search.Length);
    }

    public static bool IsTimeOfDayBetween(this DateTime time, TimeSpan startTime, TimeSpan endTime)
    {
        if (endTime == startTime)
            return true;
        else if (endTime < startTime)
            return time.TimeOfDay <= endTime || time.TimeOfDay >= startTime;
        else
            return time.TimeOfDay >= startTime && time.TimeOfDay <= endTime;
    }

    public static string Conjoin(this string[] terms)
    {
        var builder = new StringBuilder();

        switch (terms.Length)
        {
            case 0:
                break;

            case 1:
                builder.Append(terms[0]);
                break;

            default:
                for (int i = 0; i < terms.Length - 1; i++)
                {
                    builder.Append(terms[i]);

                    if (terms.Length > 2)
                        builder.Append(", ");
                    else
                        builder.Append(" ");
                }

                builder.Append("and ");
                builder.Append(terms[terms.Length - 1]);
                break;
        }

        return builder.ToString();
    }

    public static bool IsBetween(this DateTime dateTime, DateTime from, DateTime to)
    {
        return dateTime >= from && dateTime <= to;
    }

    public static List<T>[] Partition<T>(this List<T> list, int totalPartitions)
    {
        if (list == null)
            throw new ArgumentNullException("list");

        if (totalPartitions < 1)
            throw new ArgumentOutOfRangeException("totalPartitions");

        List<T>[] partitions = new List<T>[totalPartitions];

        int maxSize = (int)Math.Ceiling(list.Count / (double)totalPartitions);
        int k = 0;

        for (int i = 0; i < partitions.Length; i++)
        {
            partitions[i] = new List<T>();

            for (int j = k; j < k + maxSize; j++)
            {
                if (j >= list.Count)
                    break;
                partitions[i].Add(list[j]);
            }

            k += maxSize;
        }

        return partitions;
    }

    public static string DeepMessage(this Exception ex)
    {
        var message = ex.Message;
        var deep = ex;

        if (ex.Message == "An error occurred while saving the entity changes. See the inner exception for details.")
            message = ex.InnerException.Message;


        if (!string.IsNullOrEmpty(message))
            message = message.Replace("{", "[").Replace("}", "]");

        return message;
    }

    public static string Remove(this string s, IEnumerable<char> chars)
    {
        return new string(s.Where(c => !chars.Contains(c)).ToArray());
    }

    public static string ClipTo(this string str, int length)
    {
        if (string.IsNullOrEmpty(str))
            return str;
        else if (str.Length <= length)
            return str;
        else
            return $"{str.Left(length)} ...";
    }

    public static string Shuffle(this string str)
    {
        var list = new SortedList<int, char>();

        foreach (var c in str)
            list.Add(rand.Next(), c);

        return new string(list.Values.ToArray());
    }

    public static string ToLowerCaseString(this bool value)
    {
        return value ? "true" : "false";
    }

    public static Dictionary<string, object> ToPropertyDictionary(this object obj)
    {
        var dictionary = new Dictionary<string, object>();
        foreach (var propertyInfo in obj.GetType().GetProperties())
            if (propertyInfo.CanRead && propertyInfo.GetIndexParameters().Length == 0)
                dictionary[propertyInfo.Name] = propertyInfo.GetValue(obj, null);
        return dictionary;
    }

    public static void AddRange<T>(this ICollection<T> destination, IEnumerable<T> source)
    {
        foreach (T item in source)
        {
            destination.Add(item);
        }
    }

    public static object ToObject(this Dictionary<string, object> dict)
    {
        IDictionary<string, object> eo = new ExpandoObject() as IDictionary<string, object>;

        foreach (KeyValuePair<string, object> kvp in dict)
            eo.Add(kvp);

        return eo;
    }

    public static string FormatExplicit(this string input, object properties)
    {
        var sb = new StringBuilder(input);

        if (properties is ExpandoObject)
        {
            var expando = properties as ExpandoObject;

            foreach (var prop in expando.ToList())
                sb.Replace("{" + prop.Key + "}", prop.Value.ToString());
        }
        else
        {
            foreach (PropertyDescriptor prop in TypeDescriptor.GetProperties(properties))
                sb.Replace("{" + prop.Name + "}", (prop.GetValue(properties) ?? string.Empty).ToString());
        }

        return sb.ToString();
    }

    public static void RemoveAll<T>(this ICollection<T> collection, Func<T, bool> predicate)
    {
        T element;

        for (int i = 0; i < collection.Count; i++)
        {
            element = collection.ElementAt(i);
            if (predicate(element))
            {
                collection.Remove(element);
                i--;
            }
        }
    }

    public static List<T> AllAfter<T>(this List<T> items, T current)
    {
        var index = items.IndexOf(current);

        if (index == -1)
            throw new ArgumentOutOfRangeException();

        if (index == items.Count() - 1)
            return new List<T>();
        else
            return items.GetRange(index + 1, items.Count - index - 1);
    }

    public static List<T> AllBefore<T>(this List<T> items, T current)
    {
        var index = items.IndexOf(current);

        if (index == -1)
            throw new ArgumentOutOfRangeException();

        if (index == 0)
            return new List<T>();
        else
            return items.GetRange(0, index);
    }

    public static T ElementAfter<T>(this IEnumerable<T> items, T current)
    {
        var index = items.IndexOf(current);

        if (index == -1)
            throw new ArgumentOutOfRangeException();

        if (index == items.Count() - 1)
            throw new IndexOutOfRangeException();

        return items.ElementAt(index + 1);
    }

    public static T ElementBefore<T>(this IEnumerable<T> items, T current)
    {
        var index = items.IndexOf(current);

        if (index == -1)
            throw new ArgumentOutOfRangeException();

        if (index == 0)
            throw new IndexOutOfRangeException();

        return items.ElementAt(index - 1);
    }

    public static bool IsFirst<T>(this IEnumerable<T> items, T item)
    {
        var index = items.IndexOf(item);

        if (index == -1)
            throw new ArgumentOutOfRangeException();

        return index == 0;
    }

    public static bool IsLast<T>(this IEnumerable<T> items, T item)
    {
        var index = items.IndexOf(item);

        if (index == -1)
            throw new ArgumentOutOfRangeException();

        return index == items.Count() - 1;
    }

    public static int FindIndex<T>(this IEnumerable<T> items, Func<T, bool> predicate)
    {
        if (items == null) throw new ArgumentNullException("items");
        if (predicate == null) throw new ArgumentNullException("predicate");

        int retVal = 0;
        foreach (var item in items)
        {
            if (predicate(item)) return retVal;
            retVal++;
        }
        return -1;
    }

    public static int IndexOf<T>(this IEnumerable<T> items, T item)
    {
        return items.FindIndex(i => EqualityComparer<T>.Default.Equals(item, i));
    }

    public static decimal RoundTo(this decimal value, int? precision = null)
    {
        return decimal.Round(value, precision ?? 0, MidpointRounding.AwayFromZero);
    }

    public static string ToPrecision(this int value, int? precision = null)
    {
        return ToPrecision((decimal)value, precision);
    }

    public static string ToPrecision(this decimal value, int? precision = null)
    {
        if (precision == null || precision.Value == 0)
            return $"{value:N0}";
        else
        {
            var format = "0." + new String('0', precision.Value);
            return decimal.Round(value, precision.Value, MidpointRounding.AwayFromZero).ToString(format);
        }
    }

    public static string ToPrecision(this decimal value, int minimumPrecision, int maximumPrecision)
    {
        if (maximumPrecision == 0)
            return $"{value:N0}";

        var output = value.ToPrecision(maximumPrecision);
        var decimalAt = output.IndexOf(NumberFormatInfo.CurrentInfo.NumberDecimalSeparator);

        for (var i = maximumPrecision; i > minimumPrecision; i--)
        {
            if (output.Length > decimalAt + i && output[decimalAt + i] == '0')
                output = output.Left(output.Length - 1);
        }

        return output;
    }

    public static string Left(this string inString, int characters)
    {
        if (inString.Length < characters)
            return inString;
        else
            return inString.Substring(0, characters);
    }

    public static string Right(this string inString, int characters)
    {
        if (inString.Length < characters)
            return inString;
        else
            return inString.Substring(inString.Length - characters, characters);
    }

    public static string NullIfEmpty(this string inString)
    {
        if (inString == null)
            return null;

        if (inString.Trim() == string.Empty)
            return null;
        else
            return inString;
    }

    public static byte[] HexToData(this string hexString)
    {
        if (hexString == null)
            return null;

        if (hexString.Length % 2 == 1)
            hexString = '0' + hexString;

        byte[] data = new byte[hexString.Length / 2];

        for (int i = 0; i < data.Length; i++)
            data[i] = Convert.ToByte(hexString.Substring(i * 2, 2), 16);

        return data;
    }

    public static T First<T>(this ICollection<T> col)
    {
        return col.ElementAt(0);
    }

    public static T Last<T>(this ICollection<T> col)
    {
        return col.ElementAt(col.Count - 1);
    }
}
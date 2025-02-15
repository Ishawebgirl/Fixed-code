using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Dynamic;
using System.IO;
using System.Configuration;
using Xunit;
using System.Data;

namespace Reclaim.Api.Tests;

public static class Database
{
    public static void Reset()
    {
        var file = File.ReadAllText("../../../Scripts/reset_db.sql");
        SqlHelper.ExecuteNonQuery(file);
    }

    public static IEnumerable<dynamic> GetAuthentications(string emailAddress)
    {
        var authentications = SqlHelper.ExecuteDataTableDynamic($"select aa.*, a.uniqueid from accountauthentication aa inner join account a on a.accountid = aa.accountid where emailaddress = '{Default.CustomerEmailAddress}'");

        return authentications;
    }

    public static IEnumerable<dynamic> GetLastAuthentications(int count)
    {
        var authentications = SqlHelper.ExecuteDataTableDynamic($"select top {count} aa.*, a.uniqueid from accountauthentication aa left join account a on a.accountid = aa.accountid order by authenticatedtimestamp desc");

        return authentications;
    }

    public static void ClearEmails(dynamic account)
    {
        SqlHelper.ExecuteNonQuery($"delete email from email e inner join account a on a.accountid = e.accountid where a.emailaddress = '{account.EmailAddress}'");
    }

    public static dynamic GetAccount(string emailAddress)
    {
        var account = SqlHelper.ExecuteDataTableDynamic($"select * from account where emailaddress = '{emailAddress}'").FirstOrDefault();

        return account!;
    }

    public static IEnumerable<dynamic> GetAccountPasswords(string emailAddress)
    {
        var passwords = SqlHelper.ExecuteDataTableDynamic($"select ap.* from account a inner join accountpassword ap on a.accountid = ap.accountid where emailaddress = '{emailAddress}'");

        return passwords;
    }

    public static dynamic GetLatestEmail(dynamic account)
    {
        var email = SqlHelper.ExecuteDataTableDynamic($"select top 1 * from email e inner join account a on a.accountid = e.accountid where a.emailaddress = '{account.EmailAddress}' order by e.emailid desc").FirstOrDefault();

        return email!;
    }

    public static void ValidateEmail(dynamic account, string subject)
    {
        if (ConfigurationManager.AppSettings["TestEmailEndToEnd"] == "true")
        {
            var email = SqlHelper.ExecuteDataTableDynamic($"select * from email where accountid = {account.accountID}").First();
            System.Threading.Thread.Sleep(5000);

            var receivedEmails = new GmailHelper().GetEmails(account.EmailAddress);
            Assert.True(receivedEmails.ContainsKey(email.UniqueID));
            Assert.True(receivedEmails[email.UniqueID].Subject.IndexOf(subject) >= 0);
        }
    }

    public static void SetAccountLockedOutTimestamp(string emailAddress, DateTime dateTime)
    {
        SqlHelper.ExecuteNonQuery($"update account set lockedouttimestamp = '{dateTime}' where emailaddress = '{emailAddress}'");
    }

    public static void SetAccountPasswordExpiredTimestamp(string emailAddress, DateTime dateTime)
    {
        SqlHelper.ExecuteNonQuery($"update account set passwordexpiredtimestamp = '{dateTime}' where emailaddress = '{emailAddress}'");
    }

    public static void ClearAccountPasswordChangedTimestamp(string emailAddress)
    {
        SqlHelper.ExecuteNonQuery($"update account set passwordchangedtimestamp = null where emailaddress = '{emailAddress}'");
    }

    public static void SetAccountMagicUrlToken(string emailAddress, string token, DateTime validUntil)
    {
        SqlHelper.ExecuteNonQuery($"update account set magicurltoken = '{token}', magicurlvaliduntil = '{validUntil}' where emailaddress = '{emailAddress}'");
    }

    public static void SetAccountEmailAddressConfirmedTimestamp(string emailAddress, DateTime emailAddressConfirmedTimestamp)
    {
        SqlHelper.ExecuteNonQuery($"update account set emailaddressconfirmedtimestamp = '{emailAddressConfirmedTimestamp}' where emailaddress = '{emailAddress}'");
    }

    public static void SetAccountTombstonedTimestamp(string emailAddress, DateTime tombstonedTimestamp)
    {
        SqlHelper.ExecuteNonQuery($"update account set tombstonedtimestamp = '{tombstonedTimestamp}' where emailaddress = '{emailAddress}'");
    }

    public static void ClearAccountEmailAddressConfirmedTimestamp(string emailAddress)
    {
        SqlHelper.ExecuteNonQuery($"update account set emailaddressconfirmedtimestamp = null where emailaddress = '{emailAddress}'");
    }

    public static void SetAccountPassword(string emailAddress, string passwordHash, string passwordSalt)
    {
        SqlHelper.ExecuteNonQuery($"update account set passwordhash = '{passwordHash}', passwordsalt ='{passwordSalt}' where emailaddress = '{emailAddress}'");
    }

    public static dynamic GetLastLogEntry()
    {
        return SqlHelper.ExecuteDataTableDynamic(@"select e.code, l.text from errorcode e inner join logentry l on l.errorcodeid = e.errorcodeid where e.code != 'EmailDeliveryFailed' order by l.logentryid desc").First();
    }
}

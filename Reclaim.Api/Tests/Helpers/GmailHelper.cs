using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Reclaim.Api.Tests;

public class EmailMetadata
{
    public string? Subject { get; set; }
    public string? Body { get; set; }
}

public class GmailHelper
{
   public Dictionary<Guid, EmailMetadata> GetEmails(string to)
    {
        var emails = new Dictionary<Guid, EmailMetadata>();
        var client = new ImapX.ImapClient("imap.gmail.com", true);

        if (client.Connect())
        {
            if (client.Login("scampbell@reclaimsiu.com", "xxxxx"))
            {
                var inbox = client.Folders.Inbox;

                foreach (var message in inbox.Search($"SINCE {DateTime.Now.ToString("dd-MMM-yyyy")}"))
                {
                    if (message != null && message.From.Address == "scampbell@reclaimsiu.com" && message.To.First().Address == to)
                    {
                        var key = "Unique identifier: ";
                        var index = message.Body.Html.IndexOf(key);

                        if (index >= 0)
                        {
                            var uniqueID = message.Body.Html.Substring(index + key.Length, 36);
                            emails.Add(Guid.Parse(uniqueID), new EmailMetadata { Subject = message.Subject, Body = message.Body.Html });
                        }
                    }
                }
            }
        }

        return emails;
    }
}

using Microsoft.EntityFrameworkCore;
using Newtonsoft.Json;
using Reclaim.Api.Model;
using RestSharp;
using System.ComponentModel.Design;
using System.Dynamic;
using System.Text.RegularExpressions;
using System.Web;

namespace Reclaim.Api.Services;

public class EmailService
{
    private readonly DatabaseContext _db;
    private readonly LogService _logService;
    private readonly MasterDataService _masterDataService;
    private readonly CacheService _cacheService;
    private readonly IHttpContextAccessor _httpContextAccessor;

    public EmailService(DatabaseContext db, LogService logService, MasterDataService masterDataService, CacheService cacheService, IHttpContextAccessor httpContextAccessor)
    {
        _db = db;
        _logService = logService;
        _masterDataService = masterDataService;
        _cacheService = cacheService;
        _httpContextAccessor = httpContextAccessor;
    }

    public async Task<int> DeliverPending()
    {
        // lock (this)
        {
            var deliveredEmailCount = 0;

            var emails = await _db.Emails
                .Include(x => x.Account)
                .Include(x => x.Template)
                .Where(x => x.Status == EmailStatus.Pending && DateTime.UtcNow > x.DeliverAfter)
                .Take(Setting.MaximumEmailBatchCount)
                .OrderBy(x => x.DeliverAfter).ToListAsync();

            if (emails.Count() == 0)
                return 0;

            foreach (var email in emails)
            {
                switch (email.TemplateID)
                {
                    case (int)EmailTemplateCode.Diagnostics:
                        await DeliverPending(email);
                        break;

                    default:
                        await _logService.Add($"Skipping delivery of {email.Template.Code} email, this template is not configured for queued delivery.");
                        continue;
                }

                deliveredEmailCount++;
            }

            return deliveredEmailCount;
        }
    }

    public async Task SetReceived(Guid uniqueID)
    {
        var email = await _db.Emails.Include(x => x.Account).FirstOrDefaultAsync(x => x.UniqueID == uniqueID);

        if (email != null && email.ReceivedTimestamp == null)
        {
            email.ReceivedTimestamp = DateTime.UtcNow;
            email.Account.BouncedEmailCount = 0;
            email.Account.BouncedEmailTimestamp = null;

            await _db.SaveChangesAsync();
        }
    }

    public async Task Queue(EmailTemplateCode templateName, int accountID, DateTime deliverAfter)
    {
        var email = new Email { AccountID = accountID, TemplateID = (int)templateName, DeliverAfter = deliverAfter, Status = EmailStatus.Pending, UniqueID = Guid.NewGuid() };

        _db.Emails.Add(email);
        await _db.SaveChangesAsync();
    }

    public async Task Queue(EmailTemplateCode templateName, Account account, DateTime deliverAfter)
    {
        await Queue(templateName, account.ID, deliverAfter);
    }

    public async Task SendUnhandledException(ApiException ex)
    {
        var account = await _db.Accounts.FirstAsync(x => x.ID == Setting.SystemAdministratorAccountID);
        var dict = new Dictionary<string, string>();

        dict["Message"] = ex.Message;
        dict["Error code"] = ex.ErrorCode.ToString();
        dict["Account"] = ex.LogEntry?.Account?.EmailAddress ?? "Unknown";
        dict["IP address"] = _httpContextAccessor.HttpContext.Connection.RemoteIpAddress.ToString() ?? "Unknown";
        dict["Log entry"] = ex.LogEntry?.ID.ToString() ?? "Unknown";
        dict["Stack trace"] = ex.StackTrace ?? "Unknown";

        var templateData = new ExpandoObject().Initialize(new
        {
            tableRows = dict.Select(kvp => new { key = kvp.Key, value = kvp.Value }).ToArray()
        });

        await DeliverNow(account, EmailTemplateCode.UnhandledException, templateData);
    }	

    public async Task SendAccountConfirmation(Account account, string name, bool isPasswordRequired)
    {
        if (!isPasswordRequired)
        {
            var templateData = new ExpandoObject().Initialize(new
            {
                name,
                actionButtonUrl = $"{Setting.WebsiteRootUrl}/confirmaccount?emailAddress={HttpUtility.UrlEncode(account.EmailAddress)}&token={account.MagicUrlToken}",
            });

            await DeliverNow(account, EmailTemplateCode.ConfirmAccount, templateData);
        }
        else
        {
            var templateData = new ExpandoObject().Initialize(new
            {
                name,
                actionButtonUrl = $"{Setting.WebsiteRootUrl}/setpassword?emailAddress={HttpUtility.UrlEncode(account.EmailAddress)}&token={account.MagicUrlToken}&isAccountConfirmed=false",
            });

            await DeliverNow(account, EmailTemplateCode.ConfirmAccount, templateData);
        }
    }

    public async Task SendEmailAddressAlreadyInUse(Account account)
    {
        var templateData = new ExpandoObject().Initialize(new
        {
            actionButtonUrl = $"{Setting.WebsiteRootUrl}/signin?emailAddress={HttpUtility.UrlEncode(account.EmailAddress)}"
        });

        await DeliverNow(account, EmailTemplateCode.EmailAddressAlreadyInUse, templateData);
    }

    public async Task SendRequestPasswordReset(Account account)
    {
        var templateData = new ExpandoObject().Initialize(new
        {
            actionButtonUrl = $"{Setting.WebsiteRootUrl}/setpassword?emailAddress={HttpUtility.UrlEncode(account.EmailAddress)}&token={account.MagicUrlToken}",
        });

        await DeliverNow(account, EmailTemplateCode.RequestPasswordReset, templateData);
    }

    public async Task SendRequestPasswordResetGoogle(Account account)
    {
        var templateData = new ExpandoObject().Initialize(new
        {
        });

        await DeliverNow(account, EmailTemplateCode.RequestPasswordResetGoogle, templateData);
    }

    public async Task SendPasswordChangeConfirmation(Account account)
    {
        var templateData = new ExpandoObject().Initialize(new
        {
        });

        await DeliverNow(account, EmailTemplateCode.PasswordChangeConfirmation, templateData);
    }

    private dynamic GetDynamicTemplateData(EmailTemplate emailTemplate, Account account, Guid uniqueID)
    {
        dynamic templateData = new ExpandoObject().Initialize(new
        {
            emailAddress = account.EmailAddress,
            uniqueID,
            websiteRootUrl = Setting.WebsiteRootUrl,
            emailReceivedUrl = $"{Setting.ApiRootUrl}/content/emails/{uniqueID}/received.png",
            emailTemplateImageUrl = $"{Setting.WebsiteRootUrl}/images/logo-email.png",
            accountUniqueID = account.UniqueID,
            subject = emailTemplate.Subject,
            preheader = emailTemplate.Preheader,
            body = emailTemplate.Body,
            highlightColor = emailTemplate.HighlightColor,
            actionButtonText = emailTemplate.ActionButtonText
        });

        return templateData;
    }

    private string ReplaceTemplateVariables(string template, dynamic templateData)
    {
        var regex = new Regex(@"{{(.*?)}}");
        var matches = regex.Matches(template);

        foreach (Match match in matches)
        {
            var propertyName = match.Groups[1].Value;
            var propertyValue = ((IDictionary<string, object>)templateData).TryGetValue(propertyName, out var value) ? value?.ToString() ?? string.Empty : string.Empty;
            template = template.Replace(match.Value, propertyValue);
        }

        return template;
    }

    private async Task<bool> DeliverPending(Email email)
    {
        var templateData = GetDynamicTemplateData(email.Template, email.Account, email.UniqueID);   

        switch ((EmailTemplateCode)email.TemplateID)
        {
            case EmailTemplateCode.Diagnostics:
               
                var issues = new StatusService(_db, _cacheService).GetIssues();

                if (issues.Count > 0)
                {
                    templateData.subject = "DIAGNOSTICS FAILED";
                    templateData.body = string.Join("\r\n", issues);
                }

                break;
        }

        return await Deliver(email, email.Account, templateData);
    }

    private async Task<bool> DeliverNow(Account account, EmailTemplateCode templateName, dynamic additionalData)
    {
        var uniqueID = Guid.NewGuid();
        var template = await _db.EmailTemplates.FirstAsync(x => x.Code == templateName.ToString());
        var templateData = GetDynamicTemplateData(template, account, uniqueID);

        foreach (var item in (IDictionary<string, object>)additionalData)
            ((IDictionary<string, object>)templateData)[item.Key] = item.Value;

        var email = new Email { AccountID = account.ID, TemplateID = (int)templateName, UniqueID = uniqueID, Status = EmailStatus.Pending };

        _db.Emails.Add(email);
        await _db.SaveChangesAsync();

        templateData.subject = ReplaceTemplateVariables(templateData.subject, templateData);
        templateData.preheader = ReplaceTemplateVariables(templateData.preheader, templateData);
        templateData.body = ReplaceTemplateVariables(templateData.body, templateData);
    
        return await Deliver(email, account, templateData);
    }

#pragma warning disable CS1998 // Async method lacks 'await' operators and will run synchronously
    private async Task<bool> Deliver(Email email, Account account, dynamic templateData)
#pragma warning restore CS1998 // Async method lacks 'await' operators and will run synchronously
    {
        var task = Task.Factory
           .StartNew(() => DeliverAsTask(account.EmailAddress, email.Template.ExternalID, templateData))
           .ContinueWith(x =>
           {
               using (var db = new DatabaseContext(DbContextOptions.Get()))  // request-scoped dbcontext is already gone by the time this task executes
               {
                   var reloaded = db.Emails.First(x => x.ID == email.ID);
                   reloaded.DeliveredTimestamp = _db.ContextTimestamp;
                   reloaded.Status = x.Exception == null ? EmailStatus.Delivered : EmailStatus.Failed;

                   db.SaveChanges();

                   if (x.Exception != null)
                     throw new ApiException(ErrorCode.EmailDeliveryFailed, x.Exception.InnerException?.Message ?? x.Exception.Message);                   
               }
           });

        return true;
    }

    private bool DeliverAsTask(string toEmailAddress, string templateExternalID, dynamic templateData)
    {
        var client = new RestClient();
        var url = Setting.SendGridApiUrl;
        var method = Method.Post;
        var request = new RestRequest(url, method);
        var body = GetBody(toEmailAddress, templateExternalID, templateData);

        request.AddHeader("Authorization", $"Bearer {Setting.SendGridApiKey}");
        request.AddHeader("Content-Type", "application/json");
        request.AddBody(body as string);

        var response = client.Execute(request);
        var content = response.Content;

        var errorPrefix = $"Failed to deliver {templateExternalID} email to {toEmailAddress}";

        if (content == null)
            throw new Exception($"{errorPrefix}, no response was received from the remote server");

        if ((int)response.ResponseStatus != 1)
        {
            var message = JsonConvert.DeserializeObject<dynamic>(content).errors[0].message.ToString();
            throw new Exception($"{errorPrefix}, {message}");
        }

        return true;
    }

    private string GetBody(string toEmailAddress, string templateExternalID, dynamic templateData)
    {
        var json = new
        {
            personalizations = new[]
            {
                new
                {
                    to = new[]
                    {
                        new
                        {
                            email = toEmailAddress
                        }

                    },
                    dynamic_template_data = templateData
                }
            },
            from = new
            {
                email = Setting.SendGridFromEmailAddress,
                name = Setting.SendGridFromName
            },
            template_id = templateExternalID
        };

        var output = JsonConvert.SerializeObject(json);

        return output;
    }
}

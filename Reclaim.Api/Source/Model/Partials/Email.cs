namespace Reclaim.Api.Model;

public enum EmailTemplateCode
{
    ConfirmAccount = 1,
    EmailAddressAlreadyInUse = 2,
    PasswordChangeConfirmation = 3,
    RequestPasswordReset = 4,
    UnhandledException = 5,
    Diagnostics = 6,
    RequestPasswordResetGoogle = 7
}

public partial class Email
{
    public EmailStatus CalculateStatus()
    {
        if (this.TombstonedTimestamp != null)
            return EmailStatus.Tombstoned;
        else if (this.FailedTimestamp != null)
            return EmailStatus.Failed;
        else if (this.DeliveredTimestamp != null)
            return EmailStatus.Delivered;
        else
            return EmailStatus.Pending;
    }
}
namespace Reclaim.Api;

public static class Constant
{
    public const string BlankValue = "-";
    public const string EntityFrameworkUpdateExceptionWrapper = "An error occurred while updating the entries. See the inner exception for details.";
    public const string EntityFrameworkInsertExceptionWrapper = "An error occurred while saving the entity changes. See the inner exception for details.";
    public const string TargetInvokationExceptionWrapper = "Exception has been thrown by the target of an invocation.";
    public static readonly char[] Vowels = new char[] { 'a', 'e', 'i', 'o', 'u' };
    public const int SettingCacheTimeout = 10800;
    public const int DefaultObjectCacheTimeout = 10800;
    public const string ConnectionString = "RECLAIM_API_CONNECTION_STRING";
}

public static class RegularExpression
{
    public const string Telephone = @"\+1 \d{3}-\d{3}-\d{4}";
    public const string PostalCode = @"^(\d{5}-\d{4}|\d{5})$";
    public const string CustomerCode = @"^[A-Z0-9]{2,10}$";
    public const string SocialSecurityNumber = @"\d{3}-\d{2}-\d{4}";
    public const string EmailAddress = @"^(([^<>()[\]\\.,;:\s@\""]+(\.[^<>()[\]\\.,;:\s@\""]+)*)|(\"".+\""))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-ZÀÈÌÒÙàèìòùÁÉÍÓÚÝáéíóúýÂÊÎÔÛâêîôûÃÑÕãñõÄËÏÖÜŸäëïöüŸ¡¿çÇŒœßØøÅåÆæÞþÐð\-0-9]+\.)+[a-zA-ZÀÈÌÒÙàèìòùÁÉÍÓÚÝáéíóúýÂÊÎÔÛâêîôûÃÑÕãñõÄËÏÖÜŸäëïöüŸ¡¿çÇŒœßØøÅåÆæÞþÐð]{2,}))$";
    //public const string EmailAddress = @"^[^@]+@[^@]+\.[^@]+$";
    public const string WebSite = @"^((http|https):\/\/)?([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}$";
    public const string Url = @"((([A-Za-z]{3,9}:(?:\/\/)?)(?:[-;:&=\+\$,\w]+@)?[A-Za-z0-9.-]+|(?:www.|[-;:&=\+\$,\w]+@)[A-Za-z0-9.-]+)((?:\/[\+~%\/.\w-_]*)?\??(?:[-\+=&;%@.\w_]*)#?(?:[\w]*))?)";
    public const string Password = @"^.*(?=.{8,})(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[!*@#$%^&+=\(\)\[\]]).*$";
    public const string PascalCase = @"([A-Z])(?<=[a-z]\1|[A-Za-z]\1(?=[a-z]))";
    public const string Number = @"\d+";
}

public static class CookieName
{
    public const string AccessToken = "X-Reclaim-Access-Token";
    public const string RefreshToken = "X-Reclaim-Refresh-Token";
}
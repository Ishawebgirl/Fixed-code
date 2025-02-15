namespace Reclaim.Api;

[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
internal sealed class RequestBodyAsFormAttribute : Attribute
{
}
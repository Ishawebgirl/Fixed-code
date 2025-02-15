using Microsoft.AspNetCore.Mvc.Filters;
using Microsoft.AspNetCore.Mvc.ModelBinding;
using Reclaim.Api.Model;
using System.Diagnostics;

namespace Reclaim.Api;

[AttributeUsage(AttributeTargets.Class)]
internal class ValidateModelAttribute : ActionFilterAttribute
{
    [DebuggerHidden]
    public override void OnActionExecuting(ActionExecutingContext context)
    {
        if (context.ModelState != null)
            Validate(context.ModelState);

        base.OnActionExecuting(context);
    }

    [DebuggerHidden]
    private void Validate(ModelStateDictionary modelState)
    {
        if (!modelState.IsValid)
        {
            var errors = modelState.Values.SelectMany(v => v.Errors);
            var fields = modelState.Where(x => x.Value.ValidationState == ModelValidationState.Invalid).Select(x => x.Key).ToList();

            if (errors.Count() == 0)
                throw new ApiException(ErrorCode.ModelValidationFailed, $"The entity was missing at least one required parameter.");
            else
            {
                var output = errors.Select(x => x.ErrorMessage.Replace(" field ", " ").Replace("The ", "").Replace(".", "")).ToList();

                throw new ApiException(ErrorCode.ModelValidationFailed, $"The entity failed at least one validation test: {string.Join(", ", output)}.", fields);
            }
        }
    }
}

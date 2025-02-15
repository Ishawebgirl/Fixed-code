namespace Reclaim.Api;

using iText.Layout.Splitting;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Authorization.Infrastructure;
using Microsoft.OpenApi.Models;
using NSwag.Generation.Processors;
using NSwag.Generation.Processors.Contexts;
using Swashbuckle.AspNetCore.SwaggerGen;

public class SwaggerRoleAttributeFilter : IOperationProcessor
{
    public bool Process(OperationProcessorContext context)
    {
        var requireRoleAttribute = context.MethodInfo.GetCustomAttributes(typeof(RequireRoleAttribute), false);
        var allowAnonymousAttribute = context.MethodInfo.GetCustomAttributes(typeof(AllowAnonymousAttribute), false);
        var value = "";

        if (allowAnonymousAttribute != null && allowAnonymousAttribute.Length > 0)
            value = "Anonymous access allowed";
        else if (requireRoleAttribute != null && requireRoleAttribute.Length > 0)
        {
            var roles = ((RequireRoleAttribute)requireRoleAttribute[0]).GetRoles();

            switch (roles?.Length ?? 0)
            {
                case 0:
                    value = "Requires an authenticated account with any role";
                    break;
                case 1:
                    value = $"Requires an authenticated account with the {roles.First()} role";
                    break;
                case 2:
                    value = $"Requires an authenticated account with either the {roles.First()}, or {roles.Last()} role";
                    break;
                default:
                    value = $"Requires an authenticated account with either the {string.Join(", ", roles.Take(roles.Length - 1))}, or {roles.Last()} role";
                    break;
            }

        }

        if (value != "")
            context.OperationDescription.Operation.Description += $"<br><br><b>{value}</b>";
        else 
            throw new ApiException(Model.ErrorCode.MethodAttributeMissing, $"Method {context.MethodInfo.Name} in {context.MethodInfo.DeclaringType.Name} missing a required attribute.");
        
        return true;
    }
}



/*
        if (context.MethodInfo.GetCustomAttributes(typeof(RequireRoleAttribute), false)?.FirstOrDefault() is RequireRoleAttribute attribute)
        {
            var param = new OpenApiParameter
            {
                Name = "Authorization",
                Description = "Bearer + access token",
                Required = false,
                In = ParameterLocation.Header,
                Schema = new OpenApiSchema
                {
                    Type = "string"
                }
            };

            if (operation.Parameters == null)
                operation.Parameters = new List<OpenApiParameter> { param };
            else
                operation.Parameters.Add(param);

            operation.Description += "xxxxx";
            
        }
    }
}
*/
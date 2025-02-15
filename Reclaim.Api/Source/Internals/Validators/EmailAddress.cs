using System.ComponentModel.DataAnnotations;
using System.Text.RegularExpressions;

namespace Reclaim.Api.Dtos;

public class EmailAddress : ValidationAttribute
{
    protected override ValidationResult IsValid(object? value, ValidationContext validationContext)
    {
        if (value == null)
            return ValidationResult.Success;

        if (value.ToString()!.Length == 0)
            return new ValidationResult($"{validationContext.MemberName} cannot be empty");

        if (!Regex.Match(value.ToString()!, RegularExpression.EmailAddress).Success)
            return new ValidationResult($"{validationContext.MemberName} is not a valid email address");

        return ValidationResult.Success;
    }
}

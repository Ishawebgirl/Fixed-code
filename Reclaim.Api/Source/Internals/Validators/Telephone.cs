using System.ComponentModel.DataAnnotations;
using System.Text.RegularExpressions;

namespace Reclaim.Api.Dtos;

public class Telephone : ValidationAttribute
{
    protected override ValidationResult? IsValid(object? value, ValidationContext validationContext)
    {
        if (value == null)
            return ValidationResult.Success; 

        if (value.ToString().Length == 0)
            return new ValidationResult($"Field {validationContext.MemberName} cannot be empty");

        if (!Regex.Match(value.ToString(), RegularExpression.Telephone).Success)
            return new ValidationResult($"{validationContext.MemberName} is not a valid telephone number (+1 210-543-9876)");

        return ValidationResult.Success;
    }
}

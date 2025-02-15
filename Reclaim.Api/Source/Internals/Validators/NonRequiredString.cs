using System.ComponentModel.DataAnnotations;

namespace Reclaim.Api.Dtos;

public class NonRequiredString : ValidationAttribute
{
    private int _maxLength;

    public NonRequiredString()
    {
        _maxLength = 0;
    }

    public NonRequiredString(int maxLength)
    {
        _maxLength = maxLength;
    }

    protected override ValidationResult IsValid(object? value, ValidationContext validationContext)
    {
        if (value == null)
            return ValidationResult.Success;

        if (value.ToString()!.Length == 0)
            return ValidationResult.Success;

        if (value.ToString()!.Trim().Length > _maxLength)
            return new ValidationResult($"{validationContext.MemberName} cannot be longer than {_maxLength} characters");

        if (string.IsNullOrWhiteSpace(value.ToString()))
            return new ValidationResult($"{validationContext.MemberName} cannot be all whitespace");

        return ValidationResult.Success;
    }
}

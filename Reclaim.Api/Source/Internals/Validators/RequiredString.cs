using System.ComponentModel.DataAnnotations;

namespace Reclaim.Api.Dtos;

public class RequiredString : ValidationAttribute
{
    private int _maxLength;

    public RequiredString()
    {
        _maxLength = int.MaxValue;
    }

    public RequiredString(int maxLength)
    {
        _maxLength = maxLength;
    }

    protected override ValidationResult IsValid(object? value, ValidationContext validationContext)
    {
        if (value == null)
            return null; // already appears to be tested

        if (value.ToString()!.Length == 0)
            return new ValidationResult($"{validationContext.MemberName} cannot be empty");

        if (value.ToString().Trim().Length > _maxLength)
            return new ValidationResult($"{validationContext.MemberName} cannot be longer than {_maxLength} characters");

        if (string.IsNullOrWhiteSpace(value.ToString()))
            return new ValidationResult($"{validationContext.MemberName} cannot be all whitespace");

        return null;
    }
}

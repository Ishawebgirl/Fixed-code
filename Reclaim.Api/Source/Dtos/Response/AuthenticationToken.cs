using System.ComponentModel.DataAnnotations;
using System.Diagnostics.CodeAnalysis;

namespace Reclaim.Api.Dtos;

public record AuthenticationToken : Base
{
    /// <example>eyJhbGciOiJSUzI1NiIsImtpZCI6IjZmNzI1NDEwMW ...</example>
    [Required]
    public string AccessToken { get; set; }
    
    /// <example>bf973a98-d41d-4a7c-a434-bb14c5e55b19</example>
    [Required]
    public string RefreshToken { get; set; }
    
    /// <example>2024-10-29T20:37:04Z</example>
    [Required]
    public DateTime ValidUntil { get; set; }

    /// <example>Customer</example>
    [Required]
    public Model.Role Role { get; set; }

    /// <example>customer@test.com</example>
    [Required]
    public string EmailAddress { get; set; }

    /// <example>/account/0159B3F2-1BEB-44DD-8DFA-70F6FBB4951B/avatare.jpg</example>
    [Required]
    public string AvatarUrl { get; set; }

    /// <example>Rex McCrie</example>
    [Required]
    public string NiceName { get; set; }
}
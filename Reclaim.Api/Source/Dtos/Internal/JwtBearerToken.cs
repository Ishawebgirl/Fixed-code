namespace Reclaim.Api.Dtos
{
    public record JwtBearerToken : Base
    {
        public string role { get; set; }
        public string accessToken { get; set; }
        public string refreshToken { get; set; }
        public int expiresAt { get; set; }
    }
}
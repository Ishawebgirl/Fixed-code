namespace Reclaim.Api.Dtos
{
    public record EnumItem
    {
        public int ID { get; set; }
        public string Code { get; set; }
        public string Description { get; set; }
    }

    public record EnumData
    {
        public string Name { get; set; }
        public List<EnumItem> Items { get; set; }

        public static EnumData Convert<T>() where T : struct, IConvertible
        {
            if (!typeof(T).IsEnum)
                throw new ArgumentException("T must be an enumerated type");

            var items = new List<EnumItem>();

            foreach (var e in Enum.GetValues(typeof(T)))
                items.Add(new EnumItem { ID = (int)e, Code = e.ToString(), Description = Extensions.DisplayName((Enum)e) });

            return new EnumData { Name = typeof(T).Name, Items = items };
        }
    }
}
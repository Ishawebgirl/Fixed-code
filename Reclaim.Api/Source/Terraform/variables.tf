variable "environments" {
    default = ["dev", "staging", "prod"]
}
variable "location" {
  type = string
  default = "centralus"
}
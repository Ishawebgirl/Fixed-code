terraform {
    backend "azurerm" {
        resource_group_name  = "tfstate"
        storage_account_name = "reclaimtfstate"
        container_name       = "tfstate"
        key                  = "reclaim.tfstate" 
    }
}
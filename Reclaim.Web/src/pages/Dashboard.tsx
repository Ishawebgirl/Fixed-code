import { useEffect, useState } from 'react'
import InvestigatorDashboard from './roles/investigator/Dashboard'
import CustomerDashboard from './roles/customer/Dashboard'
import AdministratorDashboard from './roles/administrator/Dashboard'
import useIdentity from 'hooks/useIdentity'

const Dashboard = () => {
  const [role, setRole] = useState<string>()
  const identity = useIdentity()

  useEffect(() => {
    if (identity !== null) {
      setRole(identity?.role)
    }
  }, [identity])

  return role === 'Administrator' ? (
    <AdministratorDashboard />
  ) : role === 'Customer' ? (
    <CustomerDashboard />
  ) : role === 'Investigator' ? (
    <InvestigatorDashboard />
  ) : (
    <div></div>
  )
}

export default Dashboard

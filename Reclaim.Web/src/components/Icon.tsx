import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { fas } from '@fortawesome/free-solid-svg-icons'
import { far } from '@fortawesome/free-regular-svg-icons'
import { library } from '@fortawesome/fontawesome-svg-core'
library.add(fas, far)

interface IIcon {
  name: any
  className?: string
  toolTip?: string
  onClick?: any
  children?: any
}

const notSolidIcons = [
  'Claims',
  'Clipboard',
  'SquarePlus',
  'FileWord',
  'FilePdf',
  'FileImage',
  'FileExcel',
  'FileVideo',
  'FileImage',
  'Bell',
  'ChessKing',
  'FileLines',
  'PaperPlane',
  'TrashCan',
]

const Icon = ({ name, className = '', toolTip, onClick, children }: IIcon) => {
  const isNotSolid = notSolidIcons.includes(name)

  return (
    <span className={`icon ${className} icon-${name}`}>
      <FontAwesomeIcon
        title={toolTip}
        icon={isNotSolid ? far['fa' + name] : fas['fa' + name]}
        onClick={onClick}
      />
      {children}
    </span>
  )
}

export default Icon
